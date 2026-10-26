// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//! Stateless Modular prediction, directly translated from libjxl baseline
//! `b866b3defec13494bfbaf705db93581cda118e37`:
//!
//! - `lib/jxl/modular/encoding/context_predict.h`: `ClampedGradient`, `Select`,
//!   `detail::PredictOne`, `PredictorHasWeighted`, and the neighbor loads in
//!   `detail::Predict` with `kNoEdgeCases` unset.
//! - `lib/jxl/modular/options.h`: `Predictor` IDs and predictor counts.
//! - `lib/jxl/modular/modular_image.h`: `pixel_type` / `pixel_type_w` domains.
//!
//! Samples are `i32`, matching `pixel_type`; arithmetic and predictions are
//! `i64`, matching `pixel_type_w`. The type boundary excludes unsupported wider
//! samples without narrowing or lossy clamping. All stateless intermediates fit
//! in `i64` even at the `i32` extremes. Signed division truncates toward zero:
//! **do not replace `/ 2` or `/ 16` with signed right shifts**, which round
//! negative values down in Rust. Gradient alone intentionally wraps its 32-bit
//! intermediate, exactly as the C++ unsigned arithmetic does.
//!
//! `Weighted` only forwards an explicitly supplied, already rounded prediction
//! from `weighted::State::Predict`. That state, its error updates, properties,
//! MA trees, encoder predictor search, and a complete codec are not ported here.

#![forbid(unsafe_code)]

use std::fmt;

/// `options.h::Predictor`, including the two encoder-only modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Predictor {
    Zero = 0,
    Left = 1,
    Top = 2,
    Average0 = 3,
    Select = 4,
    Gradient = 5,
    Weighted = 6,
    TopRight = 7,
    TopLeft = 8,
    LeftLeft = 9,
    Average1 = 10,
    Average2 = 11,
    Average3 = 12,
    Average4 = 13,
    Best = 14,
    Variable = 15,
}

/// `options.h::kNumModularPredictors` (bitstream predictors only).
pub const NUM_MODULAR_PREDICTORS: usize = Predictor::Average4 as usize + 1;
/// `options.h::kNumModularEncoderPredictors` (includes search modes).
pub const NUM_MODULAR_ENCODER_PREDICTORS: usize = Predictor::Variable as usize + 1;
/// `options.h::kUndefinedPredictor`, a sentinel, not a valid predictor ID.
pub const UNDEFINED_PREDICTOR_ID: u32 = u32::MAX;

impl Predictor {
    /// Direct port of `PredictorHasWeighted`; search modes also require state.
    pub const fn has_weighted(self) -> bool {
        matches!(self, Self::Weighted | Self::Best | Self::Variable)
    }
}

impl TryFrom<u32> for Predictor {
    type Error = PredictorError;

    fn try_from(id: u32) -> Result<Self, Self::Error> {
        Ok(match id {
            0 => Self::Zero,
            1 => Self::Left,
            2 => Self::Top,
            3 => Self::Average0,
            4 => Self::Select,
            5 => Self::Gradient,
            6 => Self::Weighted,
            7 => Self::TopRight,
            8 => Self::TopLeft,
            9 => Self::LeftLeft,
            10 => Self::Average1,
            11 => Self::Average2,
            12 => Self::Average3,
            13 => Self::Average4,
            14 => Self::Best,
            15 => Self::Variable,
            _ => return Err(PredictorError::InvalidPredictorId(id)),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictorError {
    InvalidPredictorId(u32),
    MissingWeightedPrediction,
    InvalidDimensions,
    PositionOutOfBounds,
    DimensionOverflow,
    BufferTooSmall,
}

impl fmt::Display for PredictorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPredictorId(id) => write!(f, "invalid Modular predictor ID: {id}"),
            Self::MissingWeightedPrediction => {
                f.write_str("Weighted requires a precomputed weighted::State::Predict result")
            }
            Self::InvalidDimensions => f.write_str("width must be positive and stride >= width"),
            Self::PositionOutOfBounds => f.write_str("x is outside the visible row"),
            Self::DimensionOverflow => f.write_str("row addressing exceeds usize"),
            Self::BufferTooSmall => f.write_str("the requested visible row is not in the buffer"),
        }
    }
}

impl std::error::Error for PredictorError {}

/// The seven sample neighbors passed to `detail::PredictOne`.
/// Fields use source names; callers supplying them directly must apply edges.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Neighbors {
    pub left: i32,
    pub top: i32,
    pub toptop: i32,
    pub topleft: i32,
    pub topright: i32,
    pub leftleft: i32,
    pub toprightright: i32,
}

/// Validated row slices for raster walks; address overflow and plane bounds
/// are checked once per row instead of once per pixel.
pub(crate) struct NeighborRow<'a> {
    current: &'a [i32],
    above: Option<&'a [i32]>,
    above_two: Option<&'a [i32]>,
}

impl<'a> NeighborRow<'a> {
    pub fn new(
        pixels: &'a [i32],
        width: usize,
        stride: usize,
        y: usize,
    ) -> Result<Self, PredictorError> {
        if width == 0 || stride < width {
            return Err(PredictorError::InvalidDimensions);
        }
        let begin = y
            .checked_mul(stride)
            .ok_or(PredictorError::DimensionOverflow)?;
        let end = begin
            .checked_add(width)
            .ok_or(PredictorError::DimensionOverflow)?;
        let current = pixels
            .get(begin..end)
            .ok_or(PredictorError::BufferTooSmall)?;
        Ok(Self {
            current,
            above: (y > 0).then(|| &pixels[begin - stride..end - stride]),
            above_two: (y > 1).then(|| &pixels[begin - stride * 2..end - stride * 2]),
        })
    }

    #[inline]
    pub fn at(&self, x: usize) -> Result<Neighbors, PredictorError> {
        if x >= self.current.len() {
            return Err(PredictorError::PositionOutOfBounds);
        }
        let left = if x != 0 {
            self.current[x - 1]
        } else {
            self.above.map_or(0, |row| row[0])
        };
        let top = self.above.map_or(left, |row| row[x]);
        let topleft = if x != 0 {
            self.above.map_or(left, |row| row[x - 1])
        } else {
            left
        };
        let topright = self
            .above
            .and_then(|row| row.get(x + 1))
            .copied()
            .unwrap_or(top);
        Ok(Neighbors {
            left,
            top,
            topleft,
            topright,
            leftleft: if x > 1 { self.current[x - 2] } else { left },
            toptop: self.above_two.map_or(top, |row| row[x]),
            toprightright: self
                .above
                .and_then(|row| row.get(x + 2))
                .copied()
                .unwrap_or(topright),
        })
    }
}

impl Neighbors {
    /// Safe equivalent of the edge-aware neighbor loads in `detail::Predict`.
    /// `pixels` starts at row zero; width/stride are measured in samples. Rows
    /// through `y` must contain their full visible width (final padding need not
    /// exist). Only earlier raster-order samples are read, never padding or the
    /// sample being predicted. The caller owns reconstruction order.
    pub fn from_plane(
        pixels: &[i32],
        width: usize,
        stride: usize,
        x: usize,
        y: usize,
    ) -> Result<Self, PredictorError> {
        if width == 0 || stride < width {
            return Err(PredictorError::InvalidDimensions);
        }
        if x >= width {
            return Err(PredictorError::PositionOutOfBounds);
        }
        let row = y
            .checked_mul(stride)
            .ok_or(PredictorError::DimensionOverflow)?;
        let end = row
            .checked_add(width)
            .ok_or(PredictorError::DimensionOverflow)?;
        if end > pixels.len() {
            return Err(PredictorError::BufferTooSmall);
        }
        // The validated row also bounds all preceding-row and left neighbors.
        let pos = row + x;
        let left = if x > 0 {
            pixels[pos - 1]
        } else if y > 0 {
            pixels[pos - stride]
        } else {
            0
        };
        let top = if y > 0 { pixels[pos - stride] } else { left };
        let topleft = if x > 0 && y > 0 {
            pixels[pos - stride - 1]
        } else {
            left
        };
        let topright = if width - x > 1 && y > 0 {
            pixels[pos - stride + 1]
        } else {
            top
        };
        let leftleft = if x > 1 { pixels[pos - 2] } else { left };
        let toptop = if y > 1 {
            pixels[pos - stride - stride]
        } else {
            top
        };
        let toprightright = if width - x > 2 && y > 0 {
            pixels[pos - stride + 2]
        } else {
            topright
        };
        Ok(Self {
            left,
            top,
            toptop,
            topleft,
            topright,
            leftleft,
            toprightright,
        })
    }
}

/// `ClampedGradient(n, w, l)`: the gradient clipped to the two neighbors.
/// Wrapping is deliberate: C++ adds/subtracts as `uint32_t`, then interprets
/// the result as `int32_t`. The two comparisons repair overflow/underflow.
#[inline]
pub fn clamped_gradient(n: i32, w: i32, l: i32) -> i32 {
    let min = n.min(w);
    let max = n.max(w);
    let grad = n.wrapping_add(w).wrapping_sub(l);
    let grad_clamp_max = if l < min { max } else { grad };
    if l > max { min } else { grad_clamp_max }
}

/// `Select(a, b, c)` on the sample domain. Equal distances select **b** (top).
#[inline]
pub fn select(a: i32, b: i32, c: i32) -> i32 {
    let p = i64::from(a) + i64::from(b) - i64::from(c);
    let pa = (p - i64::from(a)).abs();
    let pb = (p - i64::from(b)).abs();
    if pa < pb { a } else { b }
}

/// `detail::PredictOne`, restricted to the source's `i32` sample domain.
///
/// `precomputed_weighted` is the final, already rounded `pixel_type_w` result
/// of `weighted::State::Predict`, NOT its internal extra-bit `pred` accumulator.
/// It is required only for `Weighted` and is forwarded without narrowing.
/// `None` on that branch is an error, not a fake zero weighted prediction.
///
/// `Best`/`Variable` return zero, exactly like the C++ switch's default branch;
/// this does NOT implement either encoder search mode. Invalid raw IDs (including
/// the undefined sentinel) must be rejected by `Predictor::try_from` first.
/// Results are not clamped to `i32` or the image's bit depth.
#[inline]
pub fn predict_one(
    predictor: Predictor,
    neighbors: Neighbors,
    precomputed_weighted: Option<i64>,
) -> Result<i64, PredictorError> {
    let left = i64::from(neighbors.left);
    let top = i64::from(neighbors.top);
    let toptop = i64::from(neighbors.toptop);
    let topleft = i64::from(neighbors.topleft);
    let topright = i64::from(neighbors.topright);
    let leftleft = i64::from(neighbors.leftleft);
    let toprightright = i64::from(neighbors.toprightright);
    Ok(match predictor {
        Predictor::Zero => 0,
        Predictor::Left => left,
        Predictor::Top => top,
        Predictor::Select => i64::from(select(neighbors.left, neighbors.top, neighbors.topleft)),
        Predictor::Weighted => {
            return precomputed_weighted.ok_or(PredictorError::MissingWeightedPrediction);
        }
        Predictor::Gradient => i64::from(clamped_gradient(
            neighbors.left,
            neighbors.top,
            neighbors.topleft,
        )),
        Predictor::TopLeft => topleft,
        Predictor::TopRight => topright,
        Predictor::LeftLeft => leftleft,
        Predictor::Average0 => (left + top) / 2,
        Predictor::Average1 => (left + topleft) / 2,
        Predictor::Average2 => (topleft + top) / 2,
        Predictor::Average3 => (top + topright) / 2,
        Predictor::Average4 => {
            (6 * top - 2 * toptop + 7 * left + leftleft + toprightright + 3 * topright + 8) / 16
        }
        Predictor::Best | Predictor::Variable => 0,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn row_neighbors_match_pointwise_edges_and_padded_stride() {
        use super::{NeighborRow, Neighbors, PredictorError};
        for width in [1, 2, 3, 7, 32, 33, 257] {
            for padding in [0, 1, 11] {
                let stride = width + padding;
                let pixels: Vec<_> = (0..stride * 5)
                    .map(|n| (n as i32).wrapping_mul(0x5c123fa1))
                    .collect();
                for y in 0..5 {
                    let row = NeighborRow::new(&pixels, width, stride, y).unwrap();
                    for x in 0..width {
                        assert_eq!(
                            row.at(x).unwrap(),
                            Neighbors::from_plane(&pixels, width, stride, x, y).unwrap()
                        );
                    }
                    assert_eq!(row.at(width), Err(PredictorError::PositionOutOfBounds));
                }
            }
        }
        assert!(NeighborRow::new(&[], 0, 0, 0).is_err());
        assert!(NeighborRow::new(&[1; 4], 3, 2, 0).is_err());
        assert!(NeighborRow::new(&[1; 4], 3, 3, 1).is_err());
        assert!(NeighborRow::new(&[], 2, usize::MAX, 2).is_err());
    }

    use super::*;

    // Independent wide scalar formulas: no production helpers, no shifts,
    // and a mathematical clamp rather than the wrapping gradient implementation.
    fn scalar(predictor: Predictor, n: Neighbors, weighted: i64) -> i64 {
        let l = i128::from(n.left);
        let t = i128::from(n.top);
        let tt = i128::from(n.toptop);
        let tl = i128::from(n.topleft);
        let tr = i128::from(n.topright);
        let ll = i128::from(n.leftleft);
        let trr = i128::from(n.toprightright);
        let result = match predictor {
            Predictor::Zero | Predictor::Best | Predictor::Variable => 0,
            Predictor::Left => l,
            Predictor::Top => t,
            Predictor::Average0 => (l + t) / 2,
            Predictor::Select => {
                if (t - tl).abs() < (l - tl).abs() {
                    l
                } else {
                    t
                }
            }
            Predictor::Gradient => (l + t - tl).clamp(l.min(t), l.max(t)),
            Predictor::Weighted => i128::from(weighted),
            Predictor::TopRight => tr,
            Predictor::TopLeft => tl,
            Predictor::LeftLeft => ll,
            Predictor::Average1 => (l + tl) / 2,
            Predictor::Average2 => (tl + t) / 2,
            Predictor::Average3 => (t + tr) / 2,
            Predictor::Average4 => (t * 6 - tt * 2 + l * 7 + ll + trr + tr * 3 + 8) / 16,
        };
        i64::try_from(result).unwrap()
    }

    fn check_all(n: Neighbors, weighted: i64) {
        for id in 0..NUM_MODULAR_ENCODER_PREDICTORS as u32 {
            let p = Predictor::try_from(id).unwrap();
            assert_eq!(
                predict_one(p, n, Some(weighted)),
                Ok(scalar(p, n, weighted)),
                "{p:?}: {n:?}"
            );
            if p != Predictor::Weighted {
                assert_eq!(predict_one(p, n, None), Ok(scalar(p, n, weighted)));
            }
        }
    }

    #[test]
    fn ids_and_weighted_classification_match_options_h() {
        let predictors = [
            Predictor::Zero,
            Predictor::Left,
            Predictor::Top,
            Predictor::Average0,
            Predictor::Select,
            Predictor::Gradient,
            Predictor::Weighted,
            Predictor::TopRight,
            Predictor::TopLeft,
            Predictor::LeftLeft,
            Predictor::Average1,
            Predictor::Average2,
            Predictor::Average3,
            Predictor::Average4,
            Predictor::Best,
            Predictor::Variable,
        ];
        assert_eq!(NUM_MODULAR_PREDICTORS, 14);
        assert_eq!(NUM_MODULAR_ENCODER_PREDICTORS, predictors.len());
        assert_eq!(UNDEFINED_PREDICTOR_ID, 0xffff_ffff);
        for (id, p) in predictors.into_iter().enumerate() {
            assert_eq!(p as u32, id as u32);
            assert_eq!(Predictor::try_from(id as u32), Ok(p));
            assert_eq!(p.has_weighted(), [6, 14, 15].contains(&id));
        }
        for id in [16, 17, 255, 0x8000_0000, UNDEFINED_PREDICTOR_ID] {
            assert_eq!(
                Predictor::try_from(id),
                Err(PredictorError::InvalidPredictorId(id))
            );
        }
    }

    #[test]
    fn fixed_vectors_cover_every_dispatch_branch() {
        let n = Neighbors {
            left: 10,
            top: 20,
            toptop: 5,
            topleft: 15,
            topright: 30,
            leftleft: 2,
            toprightright: 40,
        };
        for (id, value) in [0, 10, 20, 15, 20, 15, -123, 30, 15, 2, 12, 17, 25, 20, 0, 0]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                predict_one(Predictor::try_from(id as u32).unwrap(), n, Some(-123)),
                Ok(value)
            );
        }
        let n = Neighbors {
            left: -5,
            top: -2,
            toptop: 7,
            topleft: -3,
            topright: 4,
            leftleft: -8,
            toprightright: 9,
        };
        for (id, value) in [0, -5, -2, -3, -5, -4, 456, 4, -3, -8, -4, -2, 1, -2, 0, 0]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                predict_one(Predictor::try_from(id as u32).unwrap(), n, Some(456)),
                Ok(value)
            );
        }
    }

    #[test]
    fn negative_division_is_not_arithmetic_shift() {
        let n = Neighbors {
            left: -1,
            top: -2,
            topleft: -2,
            topright: -1,
            ..Neighbors::default()
        };
        for p in [
            Predictor::Average0,
            Predictor::Average1,
            Predictor::Average3,
        ] {
            assert_eq!(predict_one(p, n, None), Ok(-1));
        }
        assert_eq!(
            predict_one(Predictor::Average2, Neighbors { top: -1, ..n }, None),
            Ok(-1)
        );
        assert_eq!(-3_i64 >> 1, -2);
        let n = Neighbors {
            left: -2,
            ..Neighbors::default()
        };
        assert_eq!(predict_one(Predictor::Average4, n, None), Ok(0)); // (-14 + 8) / 16
        assert_eq!(-6_i64 >> 4, -1);
        assert_eq!(
            predict_one(Predictor::Average4, Neighbors { toptop: 4, ..n }, None),
            Ok(0)
        );
        assert_eq!(
            predict_one(Predictor::Average4, Neighbors { toptop: 5, ..n }, None),
            Ok(-1)
        );
        assert_eq!(
            predict_one(
                Predictor::Average4,
                Neighbors {
                    left: 2,
                    ..Neighbors::default()
                },
                None
            ),
            Ok(1)
        );
    }

    #[test]
    fn select_ties_choose_second_argument() {
        assert_eq!(select(10, 20, 15), 20);
        assert_eq!(select(20, 10, 15), 10);
        assert_eq!(select(-9, -3, -6), -3);
        assert_eq!(select(-9, -3, -3), -9);
        assert_eq!(select(-9, -3, -9), -3);
        assert_eq!(select(i32::MIN, i32::MAX, i32::MIN), i32::MAX);
        assert_eq!(select(i32::MIN, i32::MAX, i32::MAX), i32::MIN);
    }

    #[test]
    fn gradient_wrap_is_exact_for_full_i32_domain() {
        let values = [
            i32::MIN,
            i32::MIN + 1,
            -65536,
            -32768,
            -1,
            0,
            1,
            32767,
            65535,
            i32::MAX - 1,
            i32::MAX,
        ];
        for n in values {
            for w in values {
                for l in values {
                    let expected = (i64::from(n) + i64::from(w) - i64::from(l))
                        .clamp(i64::from(n.min(w)), i64::from(n.max(w)));
                    assert_eq!(i64::from(clamped_gradient(n, w, l)), expected);
                }
            }
        }
    }

    #[test]
    fn weighted_requires_an_explicit_final_prediction() {
        let n = Neighbors::default();
        assert_eq!(
            predict_one(Predictor::Weighted, n, None),
            Err(PredictorError::MissingWeightedPrediction)
        );
        for value in [i64::MIN, -65536, -1, 0, 1, 65535, i64::MAX] {
            assert_eq!(predict_one(Predictor::Weighted, n, Some(value)), Ok(value));
        }
        assert_eq!(predict_one(Predictor::Best, n, None), Ok(0));
        assert_eq!(predict_one(Predictor::Variable, n, None), Ok(0));
    }

    #[test]
    fn signed_and_unsigned_pixel16_boundaries_are_not_clipped() {
        let values = [
            i32::MIN,
            -65536,
            -65535,
            -32769,
            -32768,
            -1,
            0,
            1,
            32767,
            32768,
            65535,
            65536,
            i32::MAX,
        ];
        for a in values {
            for b in values {
                for c in values {
                    check_all(
                        Neighbors {
                            left: a,
                            top: b,
                            toptop: c,
                            topleft: c,
                            topright: a,
                            leftleft: b,
                            toprightright: c,
                        },
                        -9876543210,
                    );
                }
            }
        }
        let n = Neighbors {
            left: 65535,
            top: 65535,
            toptop: 0,
            topleft: 0,
            topright: 65535,
            leftleft: 65535,
            toprightright: 65535,
        };
        assert_eq!(predict_one(Predictor::Average4, n, None), Ok(73727));
        let n = Neighbors {
            left: i32::MAX,
            top: i32::MAX,
            toptop: i32::MIN,
            topleft: 0,
            topright: i32::MAX,
            leftleft: i32::MAX,
            toprightright: i32::MAX,
        };
        assert_eq!(predict_one(Predictor::Average4, n, None), Ok(2684354559));
        let n = Neighbors {
            left: i32::MIN,
            top: i32::MIN,
            toptop: i32::MAX,
            topleft: 0,
            topright: i32::MIN,
            leftleft: i32::MIN,
            toprightright: i32::MIN,
        };
        assert_eq!(predict_one(Predictor::Average4, n, None), Ok(-2684354559));
    }

    #[test]
    fn random_neighbors_match_explicit_wide_scalar_formulas() {
        let mut state = 0xb866_b3de_fec1_3494_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as u32
        };
        for mode in 0..3 {
            for _ in 0..10000 {
                let mut sample = || {
                    let bits = next();
                    match mode {
                        0 => (bits & 0xffff) as i32,
                        1 => i32::from(bits as i16),
                        _ => bits as i32,
                    }
                };
                let n = Neighbors {
                    left: sample(),
                    top: sample(),
                    toptop: sample(),
                    topleft: sample(),
                    topright: sample(),
                    leftleft: sample(),
                    toprightright: sample(),
                };
                check_all(n, i64::from(next() as i32));
            }
        }
    }

    #[test]
    fn edge_neighbor_vectors_with_row_padding() {
        let pixels = [
            10,
            20,
            30,
            40,
            i32::MIN,
            i32::MAX,
            50,
            60,
            70,
            80,
            i32::MIN,
            i32::MAX,
            90,
            100,
            110,
            120,
        ];
        // Array entries follow the PredictOne argument order.
        let cases = [
            (0, 0, [0, 0, 0, 0, 0, 0, 0]),
            (1, 0, [10, 10, 10, 10, 10, 10, 10]),
            (3, 0, [30, 30, 30, 30, 30, 20, 30]),
            (0, 1, [10, 10, 10, 10, 20, 10, 30]),
            (1, 1, [50, 20, 20, 10, 30, 50, 40]),
            (2, 1, [60, 30, 30, 20, 40, 50, 40]),
            (3, 1, [70, 40, 40, 30, 40, 60, 40]),
            (0, 2, [50, 50, 10, 50, 60, 50, 70]),
            (2, 2, [100, 70, 30, 60, 80, 90, 80]),
            (3, 2, [110, 80, 40, 70, 80, 100, 80]),
        ];
        for (
            x,
            y,
            [
                left,
                top,
                toptop,
                topleft,
                topright,
                leftleft,
                toprightright,
            ],
        ) in cases
        {
            let n = Neighbors::from_plane(&pixels, 4, 6, x, y).unwrap();
            assert_eq!(
                n,
                Neighbors {
                    left,
                    top,
                    toptop,
                    topleft,
                    topright,
                    leftleft,
                    toprightright
                }
            );
            check_all(n, -17);
        }
        assert_eq!(
            Neighbors::from_plane(&[123], 1, 1, 0, 0),
            Ok(Neighbors::default())
        );
        assert_eq!(
            Neighbors::from_plane(&[123], 1, usize::MAX, 0, 0),
            Ok(Neighbors::default())
        );
        assert_eq!(
            Neighbors::from_plane(&[7, -9, 42], 1, 1, 0, 2),
            Ok(Neighbors {
                left: -9,
                top: -9,
                toptop: 7,
                topleft: -9,
                topright: -9,
                leftleft: -9,
                toprightright: -9,
            })
        );
        assert_eq!(
            Neighbors::from_plane(&[1, 2, 3, 4], 2, 2, 1, 1),
            Ok(Neighbors {
                left: 3,
                top: 2,
                toptop: 2,
                topleft: 1,
                topright: 2,
                leftleft: 3,
                toprightright: 2,
            })
        );
    }

    #[test]
    fn neighbor_address_validation_does_not_wrap() {
        assert_eq!(
            Neighbors::from_plane(&[], 0, 0, 0, 0),
            Err(PredictorError::InvalidDimensions)
        );
        assert_eq!(
            Neighbors::from_plane(&[0; 4], 3, 2, 0, 0),
            Err(PredictorError::InvalidDimensions)
        );
        for x in [2, usize::MAX] {
            assert_eq!(
                Neighbors::from_plane(&[0; 4], 2, 2, x, 0),
                Err(PredictorError::PositionOutOfBounds)
            );
        }
        assert_eq!(
            Neighbors::from_plane(&[], 1, 1, 0, 0),
            Err(PredictorError::BufferTooSmall)
        );
        assert_eq!(
            Neighbors::from_plane(&[0; 3], 2, 2, 0, 1),
            Err(PredictorError::BufferTooSmall)
        );
        assert_eq!(
            Neighbors::from_plane(&[0; 4], 2, 2, 0, 2),
            Err(PredictorError::BufferTooSmall)
        );
        assert_eq!(
            Neighbors::from_plane(&[], 1, 2, 0, usize::MAX),
            Err(PredictorError::DimensionOverflow)
        );
        assert_eq!(
            Neighbors::from_plane(&[], 1, 1, 0, usize::MAX),
            Err(PredictorError::DimensionOverflow)
        );
        assert_eq!(
            Neighbors::from_plane(&[], usize::MAX, usize::MAX, 0, 0),
            Err(PredictorError::BufferTooSmall)
        );
    }

    // Separate coordinate-based oracle, including fallback at both right edges.
    fn scalar_neighbors(rows: &[Vec<i32>], x: usize, y: usize) -> Neighbors {
        let width = rows[0].len();
        let left = match (x, y) {
            (0, 0) => 0,
            (0, _) => rows[y - 1][0],
            _ => rows[y][x - 1],
        };
        let top = if y == 0 { left } else { rows[y - 1][x] };
        let topleft = if x == 0 || y == 0 {
            left
        } else {
            rows[y - 1][x - 1]
        };
        let topright = if y == 0 {
            top
        } else {
            rows[y - 1][(x + 1).min(width - 1)]
        };
        let toprightright = if y == 0 {
            topright
        } else {
            rows[y - 1][(x + 2).min(width - 1)]
        };
        Neighbors {
            left,
            top,
            topleft,
            topright,
            toprightright,
            toptop: if y < 2 { top } else { rows[y - 2][x] },
            leftleft: if x < 2 { left } else { rows[y][x - 2] },
        }
    }

    #[test]
    fn all_edges_match_scalar_and_ignore_current_future_and_padding() {
        let mut state = 0x3494_bfba_u32;
        for width in 1..=9 {
            for height in 1..=5 {
                let rows: Vec<Vec<i32>> = (0..height)
                    .map(|_| {
                        (0..width)
                            .map(|_| {
                                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                                state as i32
                            })
                            .collect()
                    })
                    .collect();
                for padding in [0, 1, 5] {
                    let stride = width + padding;
                    let mut pixels = vec![i32::MIN; stride * (height - 1) + width];
                    for (y, row) in rows.iter().enumerate() {
                        pixels[y * stride..y * stride + width].copy_from_slice(row);
                    }
                    for y in 0..height {
                        for x in 0..width {
                            let expected = scalar_neighbors(&rows, x, y);
                            let n = Neighbors::from_plane(&pixels, width, stride, x, y).unwrap();
                            assert_eq!(
                                n, expected,
                                "{width}x{height}, ({x}, {y}), stride {stride}"
                            );
                            check_all(n, 99);
                            let mut changed = pixels.clone();
                            changed[y * stride + x..].fill(i32::MAX);
                            assert_eq!(
                                Neighbors::from_plane(&changed, width, stride, x, y),
                                Ok(expected)
                            );
                        }
                    }
                }
            }
        }
    }
}
