//! Encoder-only MA feature scan over original transformed i32 samples.
//!
//! Semantics: Rust predictor::{Neighbors, clamped_gradient, predict_one} and
//! tree properties 10/11, from libjxl b866b3defec13494bfbaf705db93581cda118e37,
//! lib/jxl/modular/encoding/context_predict.h. These native Rust kernels do not
//! use fNBLI's narrowed features. Residuals are statistical-gather differences,
//! not multiplier-coded residuals. Only complete visible 16-pixel blocks use SIMD.

use crate::modular::predictor::clamped_gradient;
use crate::{Error, Result};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Backend {
    Auto,
    Scalar,
    Avx2,
    Avx512,
}

impl Backend {
    pub(super) fn available(self) -> bool {
        match self {
            Self::Auto | Self::Scalar => true,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx512 => std::is_x86_feature_detected!("avx512f"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            Self::Avx2 | Self::Avx512 => false,
        }
    }

    fn resolve(self) -> Self {
        if self != Self::Auto {
            self
        } else if Self::Avx512.available() {
            Self::Avx512
        } else if Self::Avx2.available() {
            Self::Avx2
        } else {
            Self::Scalar
        }
    }
}

pub(super) struct Row {
    /// Property 10: vertical difference Left - TopLeft, wrapping in i32.
    pub vertical: Vec<i32>,
    /// Property 11: negative horizontal difference TopLeft - Top, wrapping in i32.
    pub horizontal: Vec<i32>,
    /// Zero, Left, Top, Gradient, in that order.
    pub residuals: [Vec<i32>; 4],
}

impl Row {
    pub(super) fn new(width: usize) -> Self {
        Self {
            horizontal: vec![0; width],
            vertical: vec![0; width],
            residuals: std::array::from_fn(|_| vec![0; width]),
        }
    }
}

/// Rows through y must contain their visible width; last-row padding is not
/// required. Every output must have at least width entries; suffixes remain
/// untouched. All validation, including explicit backend availability, precedes
/// mutation. Zero width is invalid, matching Neighbors::from_plane.
pub(super) fn fill(
    pixels: &[i32],
    width: usize,
    stride: usize,
    y: usize,
    backend: Backend,
    out: &mut Row,
) -> Result<()> {
    if width == 0 || stride < width {
        return Err(Error::InvalidInput("invalid MA scan row dimensions".into()));
    }
    let begin = y
        .checked_mul(stride)
        .ok_or_else(|| Error::InvalidInput("MA scan row address overflow".into()))?;
    let end = begin
        .checked_add(width)
        .ok_or_else(|| Error::InvalidInput("MA scan row end overflow".into()))?;
    let current = pixels
        .get(begin..end)
        .ok_or_else(|| Error::InvalidInput("MA scan row is outside the input".into()))?;
    if out.horizontal.len() < width
        || out.vertical.len() < width
        || out.residuals.iter().any(|row| row.len() < width)
    {
        return Err(Error::InvalidInput(
            "MA scan output row is too short".into(),
        ));
    }
    if !backend.available() {
        return Err(Error::InvalidInput("MA scan backend is unavailable".into()));
    }
    // Checked positive stride and the current row's end also bound this row.
    let top = (y > 0).then(|| &pixels[begin - stride..end - stride]);
    let backend = backend.resolve();
    scalar_at(current, top, 0, out);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let next = {
        // SAFETY: runtime detection above covers each kernel's sole target
        // feature. All input/output lengths are checked before the first write.
        unsafe {
            match backend {
                Backend::Avx2 => x86::avx2(current, top, out),
                Backend::Avx512 => x86::avx512(current, top, out),
                Backend::Auto | Backend::Scalar => 1,
            }
        }
    };
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let next = {
        let _ = backend;
        1
    };
    for x in next..width {
        scalar_at(current, top, x, out);
    }
    Ok(())
}

#[inline]
fn scalar_at(current: &[i32], top: Option<&[i32]>, x: usize, out: &mut Row) {
    let left = if x == 0 {
        top.map_or(0, |row| row[0])
    } else {
        current[x - 1]
    };
    let north = top.map_or(left, |row| row[x]);
    let north_west = if x == 0 {
        left
    } else {
        top.map_or(left, |row| row[x - 1])
    };
    out.vertical[x] = left.wrapping_sub(north_west);
    out.horizontal[x] = north_west.wrapping_sub(north);
    let value = i64::from(current[x]);
    for (residuals, prediction) in
        out.residuals
            .iter_mut()
            .zip([0, left, north, clamped_gradient(left, north, north_west)])
    {
        residuals[x] = (value - i64::from(prediction)) as i32;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::Row;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Requires AVX2, positive width, and all slices covering the visible row.
    /// Returns the first unprocessed column for the scalar tail.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn avx2(current: &[i32], top: Option<&[i32]>, out: &mut Row) -> usize {
        let mut x = 1;
        // SAFETY: each iteration covers x..x+16, with x >= 1. Both unaligned
        // eight-lane loads/stores stay inside visible slices; shifted neighbor
        // loads start at x-1. No padding or future row is accessed.
        unsafe {
            while current.len() - x >= 16 {
                for offset in [0, 8] {
                    let i = x + offset;
                    let value = _mm256_loadu_si256(current.as_ptr().add(i).cast());
                    let left = _mm256_loadu_si256(current.as_ptr().add(i - 1).cast());
                    let (north, north_west) = if let Some(top) = top {
                        (
                            _mm256_loadu_si256(top.as_ptr().add(i).cast()),
                            _mm256_loadu_si256(top.as_ptr().add(i - 1).cast()),
                        )
                    } else {
                        (left, left)
                    };
                    let low = _mm256_min_epi32(left, north);
                    let high = _mm256_max_epi32(left, north);
                    let gradient = _mm256_sub_epi32(_mm256_add_epi32(left, north), north_west);
                    // Compare TopLeft, not the wrapped gradient. Clamping the
                    // latter would change full-range i32 overflow semantics.
                    let prediction = _mm256_blendv_epi8(
                        _mm256_blendv_epi8(gradient, high, _mm256_cmpgt_epi32(low, north_west)),
                        low,
                        _mm256_cmpgt_epi32(north_west, high),
                    );
                    _mm256_storeu_si256(
                        out.vertical.as_mut_ptr().add(i).cast(),
                        _mm256_sub_epi32(left, north_west),
                    );
                    _mm256_storeu_si256(
                        out.horizontal.as_mut_ptr().add(i).cast(),
                        _mm256_sub_epi32(north_west, north),
                    );
                    for (residuals, residual) in out.residuals.iter_mut().zip([
                        value,
                        _mm256_sub_epi32(value, left),
                        _mm256_sub_epi32(value, north),
                        _mm256_sub_epi32(value, prediction),
                    ]) {
                        _mm256_storeu_si256(residuals.as_mut_ptr().add(i).cast(), residual);
                    }
                }
                x += 16;
            }
        }
        x
    }

    /// Requires AVX-512F only, with the same slice contract as avx2.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn avx512(current: &[i32], top: Option<&[i32]>, out: &mut Row) -> usize {
        let mut x = 1;
        // SAFETY: complete sixteen-lane blocks and their x-1 neighbor loads
        // remain within the checked visible rows. Stores cover x..x+16 only.
        unsafe {
            while current.len() - x >= 16 {
                let value = _mm512_loadu_si512(current.as_ptr().add(x).cast());
                let left = _mm512_loadu_si512(current.as_ptr().add(x - 1).cast());
                let (north, north_west) = if let Some(top) = top {
                    (
                        _mm512_loadu_si512(top.as_ptr().add(x).cast()),
                        _mm512_loadu_si512(top.as_ptr().add(x - 1).cast()),
                    )
                } else {
                    (left, left)
                };
                let low = _mm512_min_epi32(left, north);
                let high = _mm512_max_epi32(left, north);
                let gradient = _mm512_sub_epi32(_mm512_add_epi32(left, north), north_west);
                let prediction = _mm512_mask_mov_epi32(
                    _mm512_mask_mov_epi32(gradient, _mm512_cmpgt_epi32_mask(low, north_west), high),
                    _mm512_cmpgt_epi32_mask(north_west, high),
                    low,
                );
                _mm512_storeu_si512(
                    out.vertical.as_mut_ptr().add(x).cast(),
                    _mm512_sub_epi32(left, north_west),
                );
                _mm512_storeu_si512(
                    out.horizontal.as_mut_ptr().add(x).cast(),
                    _mm512_sub_epi32(north_west, north),
                );
                for (residuals, residual) in out.residuals.iter_mut().zip([
                    value,
                    _mm512_sub_epi32(value, left),
                    _mm512_sub_epi32(value, north),
                    _mm512_sub_epi32(value, prediction),
                ]) {
                    _mm512_storeu_si512(residuals.as_mut_ptr().add(x).cast(), residual);
                }
                x += 16;
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::predictor::{Neighbors, Predictor, predict_one};

    const BACKENDS: [Backend; 4] = [
        Backend::Scalar,
        Backend::Auto,
        Backend::Avx2,
        Backend::Avx512,
    ];
    const SENTINEL: i32 = 0x6a5b_4c3d;

    fn snapshot(row: &Row) -> [Vec<i32>; 6] {
        [
            row.horizontal.clone(),
            row.vertical.clone(),
            row.residuals[0].clone(),
            row.residuals[1].clone(),
            row.residuals[2].clone(),
            row.residuals[3].clone(),
        ]
    }

    // Independent of scalar_at: use all public neighbor edge rules and wide
    // MA predictions before casting the statistical-gather differences.
    fn oracle(pixels: &[i32], width: usize, stride: usize, y: usize) -> Row {
        let mut expected = Row::new(width);
        for x in 0..width {
            let n = Neighbors::from_plane(pixels, width, stride, x, y).unwrap();
            expected.vertical[x] = n.left.wrapping_sub(n.topleft);
            expected.horizontal[x] = n.topleft.wrapping_sub(n.top);
            for (slot, predictor) in [
                Predictor::Zero,
                Predictor::Left,
                Predictor::Top,
                Predictor::Gradient,
            ]
            .into_iter()
            .enumerate()
            {
                expected.residuals[slot][x] = (i64::from(pixels[y * stride + x])
                    - predict_one(predictor, n, None).unwrap())
                    as i32;
            }
        }
        expected
    }

    fn check(pixels: &[i32], width: usize, stride: usize, y: usize) {
        let expected = snapshot(&oracle(pixels, width, stride, y));
        for backend in BACKENDS {
            if !backend.available() {
                continue;
            }
            // Exact-sized and oversized buffers catch omitted writes and
            // ensure the contract requires no writable tail padding.
            for suffix in [0, 3] {
                let mut actual = Row::new(width + suffix);
                actual.horizontal.fill(SENTINEL);
                actual.vertical.fill(SENTINEL);
                for residuals in &mut actual.residuals {
                    residuals.fill(SENTINEL);
                }
                fill(pixels, width, stride, y, backend, &mut actual).unwrap();
                for (slot, (got, want)) in snapshot(&actual).iter().zip(&expected).enumerate() {
                    assert_eq!(
                        &got[..width],
                        want,
                        "{backend:?}, width={width}, stride={stride}, y={y}, output={slot}"
                    );
                    assert!(got[width..].iter().all(|&v| v == SENTINEL));
                }
            }
        }
    }

    #[test]
    fn widths_edges_padded_rows_and_full_range_patterns_match_oracle() {
        let extremes = [i32::MIN, i32::MAX, -1, 0, 1, i32::MIN + 1, i32::MAX - 1];
        let mut random = 0x9371_05ab_u32;
        for width in 1..=35 {
            for height in 1..=3 {
                for padding in [0, 1, 7] {
                    let stride = width + padding;
                    for pattern in 0..5 {
                        // No last-row padding, and a deliberately unaligned
                        // input start. Intermediate padding is never a neighbor.
                        let len = (height - 1) * stride + width;
                        let mut storage = vec![SENTINEL; len + 1];
                        let pixels = &mut storage[1..];
                        for y in 0..height {
                            for x in 0..width {
                                random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                                pixels[y * stride + x] = match pattern {
                                    0 => random as i32,
                                    1 => extremes[(x + 3 * y) % extremes.len()],
                                    2 => {
                                        if (x + y) % 2 == 0 {
                                            i32::MIN
                                        } else {
                                            i32::MAX
                                        }
                                    }
                                    3 => i32::MIN,
                                    _ => i32::MAX,
                                };
                            }
                        }
                        for y in 0..height {
                            // Each call sees only rows through y, not future
                            // samples or current-row padding beyond the width.
                            check(&pixels[..y * stride + width], width, stride, y);
                        }
                        for y in 0..height - 1 {
                            pixels[y * stride + width..(y + 1) * stride].fill(!SENTINEL);
                        }
                        check(pixels, width, stride, height - 1);
                    }
                }
            }
        }
    }

    #[test]
    fn extreme_neighbor_triples_at_simd_half_block_and_tail_boundaries() {
        let values = [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX];
        let width = 35;
        let stride = 39;
        for left in values {
            for top in values {
                for top_left in values {
                    for x in [1, 8, 9, 16, 17, 24, 25, 32, 34] {
                        let mut pixels = vec![0; stride + width];
                        pixels[x - 1] = top_left;
                        pixels[x] = top;
                        pixels[stride + x - 1] = left;
                        pixels[stride + x] = i32::MIN.wrapping_add(top_left);
                        check(&pixels, width, stride, 1);
                    }
                }
            }
        }
    }

    #[test]
    fn borders_use_left_fallback_and_properties_are_not_swapped() {
        let pixels = [10, 30, 50, 7, -5, 100];
        let mut row = Row::new(3);
        for backend in BACKENDS {
            if !backend.available() {
                continue;
            }
            fill(&pixels, 3, 3, 0, backend, &mut row).unwrap();
            assert_eq!(row.horizontal, [0, 0, 0]);
            assert_eq!(row.vertical, [0, 0, 0]);
            assert_eq!(row.residuals[1], [10, 20, 20]);
            assert_eq!(row.residuals[1], row.residuals[2]);
            assert_eq!(row.residuals[1], row.residuals[3]);
            fill(&pixels, 3, 3, 1, backend, &mut row).unwrap();
            assert_eq!(row.vertical, [0, -3, -35]);
            assert_eq!(row.horizontal, [0, -20, -20]);
            assert_eq!(row.residuals[1][0], -3);
        }
        // Huge stride is legal for row zero: no unused stride addition.
        check(&[i32::MIN], 1, usize::MAX, 0);
    }

    fn rejects_unchanged(
        pixels: &[i32],
        width: usize,
        stride: usize,
        y: usize,
        backend: Backend,
        out: &mut Row,
    ) {
        let before = snapshot(out);
        assert!(matches!(
            fill(pixels, width, stride, y, backend, out),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(snapshot(out), before);
    }

    #[test]
    fn invalid_geometry_outputs_and_unavailable_backends_do_not_mutate() {
        let pixels = [1, 2, 3, 4, 5, 6];
        for backend in BACKENDS {
            let mut out = Row::new(4);
            out.horizontal.fill(SENTINEL);
            out.vertical.fill(!SENTINEL);
            for (i, residuals) in out.residuals.iter_mut().enumerate() {
                residuals.fill(i as i32 + 123);
            }
            for (width, stride, y) in [
                (0, 0, 0),
                (3, 2, 0),
                (3, 3, 2),
                (4, 4, 1),
                (1, 2, usize::MAX),
                (1, 1, usize::MAX),
                (2, usize::MAX, 1),
                (usize::MAX, usize::MAX, 0),
            ] {
                rejects_unchanged(&pixels, width, stride, y, backend, &mut out);
            }
            rejects_unchanged(&[], 1, 1, 0, backend, &mut out);
            for short_slot in 0..6 {
                let mut short = Row::new(4);
                match short_slot {
                    0 => short.horizontal.truncate(2),
                    1 => short.vertical.truncate(2),
                    _ => short.residuals[short_slot - 2].truncate(2),
                }
                rejects_unchanged(&pixels, 3, 3, 1, backend, &mut short);
            }
            if !backend.available() {
                rejects_unchanged(&pixels, 3, 3, 1, backend, &mut out);
            }
        }
        assert!(Backend::Auto.available());
        assert!(Backend::Scalar.available());
        let empty = Row::new(0);
        assert!(snapshot(&empty).iter().all(Vec::is_empty));
    }
}
