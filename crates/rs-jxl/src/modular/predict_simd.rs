// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

//! Native row residual kernels from libjxl b866b3defec13494bfbaf705db93581cda118e37:
//! `lib/jxl/modular/encoding/context_predict.h`, `ClampedGradient`, `Select`,
//! `detail::PredictOne`, and the edge-aware loads in `detail::Predict`.
//!
//! Parent integration: register this module, then call `residual_row` once per
//! channel row and stateless predictor. In `encode::tokenize`, a fixed leaf can
//! consume this row directly; a dynamic tree can select from cached predictor
//! rows after lookup. Subtract the leaf offset from the returned i64 residual,
//! check divisibility, divide by the multiplier, THEN check conversion to i32.
//! `encode::learn` can sample these same rows in candidate-predictor order.
//! Weighted state/properties must still advance in raster order when needed.
//! No parent module or encode call site is changed by this file.
//!
//! SIMD uses signed i64 lanes, including subtraction, across the ENTIRE i32
//! sample domain. No range hint or pre-scan is required for transformed 8/16-bit
//! data. Average divisions truncate toward zero, not toward negative infinity.
//! The wide mathematical gradient clamp equals the source's wrapping-i32
//! gradient and corrective comparisons, including at both i32 extremes.
//! Select ties choose top. No sample, prediction, or residual is saturated.
//!
//! Runtime AVX2/SSE2 dispatch is x86_64-only; other targets and short rows use
//! the existing safe scalar equations. Borders/tails need no readable padding.
//! Omitted: Weighted, predictor search, tree/property evaluation, reconstruction,
//! fused multi-predictor traversal, narrow-lane specialization, and non-x86 SIMD.

use super::predictor::{Neighbors, Predictor, PredictorError, predict_one};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowError {
    Plane(PredictorError),
    OutputTooSmall,
    UnsupportedPredictor(Predictor),
}

impl fmt::Display for RowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plane(error) => error.fmt(f),
            Self::OutputTooSmall => f.write_str("residual output is shorter than the visible row"),
            Self::UnsupportedPredictor(p) => {
                write!(f, "row SIMD requires a stateless predictor, got {p:?}")
            }
        }
    }
}

impl std::error::Error for RowError {}

struct Row<'a> {
    pixels: &'a [i32],
    width: usize,
    stride: usize,
    y: usize,
    start: usize,
}

impl<'a> Row<'a> {
    fn new(
        pixels: &'a [i32],
        width: usize,
        stride: usize,
        y: usize,
        predictor: Predictor,
        output_len: usize,
    ) -> Result<Self, RowError> {
        if predictor.has_weighted() {
            return Err(RowError::UnsupportedPredictor(predictor));
        }
        // Reuse the scalar module's checked addressing and visible-row rules.
        Neighbors::from_plane(pixels, width, stride, 0, y).map_err(RowError::Plane)?;
        if output_len < width {
            return Err(RowError::OutputTooSmall);
        }
        Ok(Self {
            pixels,
            width,
            stride,
            y,
            start: y * stride,
        })
    }

    fn scalar(&self, predictor: Predictor, x: usize) -> i64 {
        let n = Neighbors::from_plane(self.pixels, self.width, self.stride, x, self.y)
            .expect("validated visible row and column");
        i64::from(self.pixels[self.start + x])
            - predict_one(predictor, n, None).expect("validated stateless predictor")
    }

    fn scalar_range(&self, predictor: Predictor, output: &mut [i64], from: usize, to: usize) {
        for (x, residual) in output.iter_mut().enumerate().take(to).skip(from) {
            *residual = self.scalar(predictor, x);
        }
    }
}

/// Write `pixel - prediction` for row `y`, leaving `output[width..]` untouched.
/// Width must be positive, stride >= width, and rows 0..=y must be visible in
/// `pixels`. Final-row padding is unnecessary. Errors leave all output intact.
/// This is an encoder operation: the current row contains original samples,
/// not residuals awaiting sequential reconstruction. No leaf offset/multiplier
/// or signed packing is applied. Weighted/Best/Variable are explicit errors.
pub fn residual_row(
    pixels: &[i32],
    width: usize,
    stride: usize,
    y: usize,
    predictor: Predictor,
    output: &mut [i64],
) -> Result<(), RowError> {
    let row = Row::new(pixels, width, stride, y, predictor, output.len())?;
    let first = width.min(2);
    row.scalar_range(predictor, output, 0, first);
    #[allow(unused_mut)]
    let mut next = first;
    #[cfg(target_arch = "x86_64")]
    {
        if width >= 8 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2/OS support detected; Row validated the full visible
            // rows and output. The backend reads only complete interior blocks.
            next = unsafe { x86::avx2_row(&row, predictor, output) };
        } else if width >= 6 && std::is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 detected and the same slice/border contract holds.
            next = unsafe { x86::sse2_row(&row, predictor, output) };
        }
    }
    row.scalar_range(predictor, output, next, width);
    Ok(())
}

/// Portable reference/fallback with exactly the same validation and output
/// contract as `residual_row`; useful for caller-side parity checks.
pub fn residual_row_scalar(
    pixels: &[i32],
    width: usize,
    stride: usize,
    y: usize,
    predictor: Predictor,
    output: &mut [i64],
) -> Result<(), RowError> {
    let row = Row::new(pixels, width, stride, y, predictor, output.len())?;
    row.scalar_range(predictor, output, 0, width);
    Ok(())
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Predictor, Row};

    // The shared equations expand to native intrinsics, without trait dispatch.
    // P is selected once per row so unused neighbor loads/branches can disappear.
    macro_rules! kernel {
        ($feature:literal, $lanes:literal, $vector:ty) => {
            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn choose(mask: $vector, yes: $vector, no: $vector) -> $vector {
                or(and(mask, yes), andnot(mask, no))
            }

            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn abs(value: $vector) -> $vector {
                // SAFETY: same target feature as the sign helper.
                let mask = unsafe { sign(value) };
                sub(xor(value, mask), mask)
            }

            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn divide<const BITS: i32>(value: $vector) -> $vector {
                // Logical shift of magnitude, followed by restoring the sign,
                // implements C++ signed division toward zero even for -1/-15.
                // SAFETY: all helpers require this function's target feature.
                unsafe {
                    let mask = sign(value);
                    let magnitude = sub(xor(value, mask), mask);
                    sub(xor(shr::<BITS>(magnitude), mask), mask)
                }
            }

            #[target_feature(enable = $feature)]
            pub(super) unsafe fn run<const P: u32>(row: &Row<'_>, out: &mut [i64]) -> usize {
                let mut x = 2;
                let end = row.width.saturating_sub(2);
                let current = &row.pixels[row.start..row.start + row.width];
                let top_start = if row.y > 0 {
                    row.start - row.stride
                } else {
                    row.start
                };
                let top = &row.pixels[top_start..top_start + row.width];
                let tt_start = if row.y > 1 {
                    top_start - row.stride
                } else {
                    top_start
                };
                let toptop = &row.pixels[tt_start..tt_start + row.width];
                // SAFETY: the caller checked CPU features, row/output lengths,
                // and a stateless P. x >= 2 and x + LANES <= width - 2 bound
                // every load at x-2..x+2, including first-row substitutions.
                // load/store use unaligned accesses of exactly LANES elements.
                unsafe {
                    while x <= end && end - x >= $lanes {
                        let l = load(current.as_ptr().add(x - 1));
                        let t = if row.y > 0 {
                            load(top.as_ptr().add(x))
                        } else {
                            l
                        };
                        let tl = if row.y > 0 {
                            load(top.as_ptr().add(x - 1))
                        } else {
                            l
                        };
                        let tr = if row.y > 0 {
                            load(top.as_ptr().add(x + 1))
                        } else {
                            t
                        };
                        let prediction = match P {
                            0 => splat(0),
                            1 => l,
                            2 => t,
                            3 => divide::<1>(add(l, t)),
                            4 => choose(gt(abs(sub(l, tl)), abs(sub(t, tl))), l, t),
                            5 => {
                                let l_gt_t = gt(l, t);
                                let low = choose(l_gt_t, t, l);
                                let high = choose(l_gt_t, l, t);
                                let g = sub(add(l, t), tl);
                                let g = choose(gt(g, high), high, g);
                                choose(gt(low, g), low, g)
                            }
                            7 => tr,
                            8 => tl,
                            9 => load(current.as_ptr().add(x - 2)),
                            10 => divide::<1>(add(l, tl)),
                            11 => divide::<1>(add(tl, t)),
                            12 => divide::<1>(add(t, tr)),
                            13 => {
                                let tt = if row.y > 1 {
                                    load(toptop.as_ptr().add(x))
                                } else {
                                    t
                                };
                                let ll = load(current.as_ptr().add(x - 2));
                                let trr = if row.y > 0 {
                                    load(top.as_ptr().add(x + 2))
                                } else {
                                    tr
                                };
                                let six_t = add(shl::<2>(t), shl::<1>(t));
                                let seven_l = sub(shl::<3>(l), l);
                                let three_tr = add(shl::<1>(tr), tr);
                                let sum = add(sub(six_t, shl::<1>(tt)), seven_l);
                                let sum = add(add(add(sum, ll), trr), three_tr);
                                divide::<4>(add(sum, splat(8)))
                            }
                            _ => unreachable!("validated stateless predictor"),
                        };
                        store(
                            out.as_mut_ptr().add(x),
                            sub(load(current.as_ptr().add(x)), prediction),
                        );
                        x += $lanes;
                    }
                }
                x
            }
        };
    }

    mod sse2 {
        use super::Row;
        use std::arch::x86_64::*;
        use std::arch::x86_64::{
            _mm_add_epi64 as add, _mm_and_si128 as and, _mm_andnot_si128 as andnot,
            _mm_or_si128 as or, _mm_set1_epi64x as splat, _mm_slli_epi64 as shl,
            _mm_srli_epi64 as shr, _mm_sub_epi64 as sub, _mm_xor_si128 as xor,
        };

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn load(ptr: *const i32) -> __m128i {
            // SAFETY: the kernel supplies exactly two readable i32 samples.
            let packed = unsafe { _mm_loadl_epi64(ptr.cast()) };
            _mm_unpacklo_epi32(packed, _mm_srai_epi32::<31>(packed))
        }

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn store(ptr: *mut i64, value: __m128i) {
            // SAFETY: the kernel supplies exactly two writable i64 samples.
            unsafe { _mm_storeu_si128(ptr.cast(), value) };
        }

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn sign(value: __m128i) -> __m128i {
            _mm_srai_epi32::<31>(_mm_shuffle_epi32::<0xf5>(value))
        }

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn gt(a: __m128i, b: __m128i) -> __m128i {
            // SSE2 has no signed i64 compare: compare signed high dwords, then
            // unsigned low dwords on equality, replicating to each whole lane.
            let ah = _mm_shuffle_epi32::<0xf5>(a);
            let bh = _mm_shuffle_epi32::<0xf5>(b);
            let bias = _mm_set1_epi32(i32::MIN);
            let low = _mm_shuffle_epi32::<0xa0>(_mm_cmpgt_epi32(xor(a, bias), xor(b, bias)));
            or(_mm_cmpgt_epi32(ah, bh), and(_mm_cmpeq_epi32(ah, bh), low))
        }

        kernel!("sse2", 2, __m128i);
    }

    mod avx2 {
        use super::Row;
        use std::arch::x86_64::*;
        use std::arch::x86_64::{
            _mm256_add_epi64 as add, _mm256_and_si256 as and, _mm256_andnot_si256 as andnot,
            _mm256_cmpgt_epi64 as gt, _mm256_or_si256 as or, _mm256_set1_epi64x as splat,
            _mm256_slli_epi64 as shl, _mm256_srli_epi64 as shr, _mm256_sub_epi64 as sub,
            _mm256_xor_si256 as xor,
        };

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn load(ptr: *const i32) -> __m256i {
            // SAFETY: the kernel supplies exactly four readable i32 samples.
            _mm256_cvtepi32_epi64(unsafe { _mm_loadu_si128(ptr.cast()) })
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn store(ptr: *mut i64, value: __m256i) {
            // SAFETY: the kernel supplies exactly four writable i64 samples.
            unsafe { _mm256_storeu_si256(ptr.cast(), value) };
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn sign(value: __m256i) -> __m256i {
            gt(splat(0), value)
        }

        kernel!("avx2", 4, __m256i);
    }

    macro_rules! dispatch {
        ($name:ident, $backend:ident, $feature:literal) => {
            #[target_feature(enable = $feature)]
            pub(super) unsafe fn $name(row: &Row<'_>, p: Predictor, out: &mut [i64]) -> usize {
                // SAFETY: propagated backend CPU/slice contract; P is fixed by
                // this match, including exclusion of all stateful/search modes.
                unsafe {
                    match p {
                        Predictor::Zero => $backend::run::<0>(row, out),
                        Predictor::Left => $backend::run::<1>(row, out),
                        Predictor::Top => $backend::run::<2>(row, out),
                        Predictor::Average0 => $backend::run::<3>(row, out),
                        Predictor::Select => $backend::run::<4>(row, out),
                        Predictor::Gradient => $backend::run::<5>(row, out),
                        Predictor::TopRight => $backend::run::<7>(row, out),
                        Predictor::TopLeft => $backend::run::<8>(row, out),
                        Predictor::LeftLeft => $backend::run::<9>(row, out),
                        Predictor::Average1 => $backend::run::<10>(row, out),
                        Predictor::Average2 => $backend::run::<11>(row, out),
                        Predictor::Average3 => $backend::run::<12>(row, out),
                        Predictor::Average4 => $backend::run::<13>(row, out),
                        _ => unreachable!("validated stateless predictor"),
                    }
                }
            }
        };
    }

    dispatch!(sse2_row, sse2, "sse2");
    dispatch!(avx2_row, avx2, "avx2");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn predictors() -> impl Iterator<Item = Predictor> {
        (0..14)
            .filter(|&id| id != 6)
            .map(|id| Predictor::try_from(id).unwrap())
    }

    fn check(pixels: &[i32], width: usize, stride: usize, y: usize, p: Predictor) {
        let mut expected = vec![i64::MIN; width + 3];
        residual_row_scalar(pixels, width, stride, y, p, &mut expected).unwrap();
        // Independently compute each point using the unchanged scalar module.
        for x in 0..width {
            let n = Neighbors::from_plane(pixels, width, stride, x, y).unwrap();
            assert_eq!(
                expected[x],
                i64::from(pixels[y * stride + x]) - predict_one(p, n, None).unwrap()
            );
        }
        let mut actual = vec![i64::MIN; width + 3];
        residual_row(pixels, width, stride, y, p, &mut actual).unwrap();
        assert_eq!(actual, expected, "dispatch {p:?}, width {width}, y {y}");
        #[cfg(target_arch = "x86_64")]
        for (supported, lanes, backend) in [
            (
                std::is_x86_feature_detected!("sse2"),
                2,
                x86::sse2_row as unsafe fn(&Row<'_>, Predictor, &mut [i64]) -> usize,
            ),
            (
                std::is_x86_feature_detected!("avx2"),
                4,
                x86::avx2_row as unsafe fn(&Row<'_>, Predictor, &mut [i64]) -> usize,
            ),
        ] {
            if !supported || width < lanes + 4 {
                continue;
            }
            actual.fill(i64::MIN);
            let row = Row::new(pixels, width, stride, y, p, actual.len()).unwrap();
            row.scalar_range(p, &mut actual, 0, 2);
            // SAFETY: runtime feature check and Row/output validation above.
            let end = unsafe { backend(&row, p, &mut actual) };
            assert!(end >= 2 + lanes, "test must execute SIMD, not only borders");
            row.scalar_range(p, &mut actual, end, width);
            assert_eq!(
                actual, expected,
                "{lanes} lanes, {p:?}, width {width}, y {y}"
            );
        }
    }

    #[test]
    fn all_predictors_rows_tails_alignments_and_transformed_domains() {
        let mut state = 0xb866_b3de_fec1_3494_u64;
        for width in (1..=35).chain([63, 64, 65, 127, 128, 129]) {
            for padding in [0, 1, 7] {
                let stride = width + padding;
                for offset in 0..4 {
                    for mode in 0..5 {
                        // No padding at the end of the last row; other padding
                        // deliberately contains values outside the small domain.
                        let len = stride * 3 + width;
                        let mut storage = vec![i32::MIN; offset + len];
                        let pixels = &mut storage[offset..];
                        for y in 0..4 {
                            for x in 0..width {
                                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                let bits = (state >> 32) as u32;
                                pixels[y * stride + x] = match mode {
                                    0 => (bits & 255) as i32,
                                    1 => (bits & 65535) as i32,
                                    2 => (bits % 262143) as i32 - 131071,
                                    3 => bits as i32,
                                    _ => [i32::MIN, i32::MAX, -1, 0, 1, -17, 17][(x + y) % 7],
                                };
                            }
                        }
                        for y in 0..4 {
                            for p in predictors() {
                                check(pixels, width, stride, y, p);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn negative_rounding_select_ties_and_gradient_wrapping() {
        // Place each interesting combination at an interior SIMD lane, rotating
        // through both backends' lanes rather than testing only scalar edges.
        let values = [i32::MIN, i32::MAX, -17, -2, -1, 0, 1, 2, 17];
        for x in 2..10 {
            for &l in &values {
                for &t in &values {
                    for &tl in &values {
                        let mut pixels = [0; 48];
                        pixels[32 + x - 1] = l;
                        pixels[16 + x] = t;
                        pixels[16 + x - 1] = tl;
                        pixels[x] = i32::MIN;
                        pixels[16 + x + 1] = -1;
                        pixels[16 + x + 2] = i32::MAX;
                        pixels[32 + x] = i32::MIN;
                        for p in [
                            Predictor::Select,
                            Predictor::Gradient,
                            Predictor::Average0,
                            Predictor::Average1,
                            Predictor::Average2,
                            Predictor::Average3,
                            Predictor::Average4,
                        ] {
                            check(&pixels, 16, 16, 2, p);
                        }
                    }
                }
            }
        }
        let pixels = [i32::MAX; 24];
        let mut output = [0; 8];
        residual_row(&pixels, 8, 8, 0, Predictor::Zero, &mut output).unwrap();
        assert_eq!(output, [i64::from(i32::MAX); 8]);
        let mut pixels = [i32::MIN; 24];
        pixels[..8].fill(i32::MAX);
        residual_row(&pixels, 8, 8, 1, Predictor::Top, &mut output).unwrap();
        assert_eq!(output, [-4_294_967_295; 8]);
    }

    #[test]
    fn wide_average4_and_unaligned_output_slices() {
        let mut pixels = [i32::MAX; 48];
        pixels[..16].fill(i32::MIN);
        pixels[34] = i32::MIN;
        check(&pixels, 16, 16, 2, Predictor::Average4);
        for offset in 0..4 {
            let mut storage = [i64::MIN; 24];
            let output = &mut storage[offset..offset + 16];
            residual_row(&pixels, 16, 16, 2, Predictor::Average4, output).unwrap();
            // Prediction = 2684354559, larger than i32::MAX. Narrowing either
            // that prediction or its residual would lose transformed samples.
            assert_eq!(output[2], -4_831_838_207);
            assert!(storage[..offset].iter().all(|&v| v == i64::MIN));
            assert!(storage[offset + 16..].iter().all(|&v| v == i64::MIN));
        }
        pixels.fill(0);
        pixels[33] = -2;
        pixels[34] = 7;
        let mut output = [0; 16];
        residual_row(&pixels, 16, 16, 2, Predictor::Average4, &mut output).unwrap();
        // (-14 + 8) / 16 = 0, whereas an arithmetic shift would give -1.
        assert_eq!(output[2], 7);
        check(&pixels, 16, 16, 2, Predictor::Average4);
    }

    #[test]
    fn invalid_requests_are_atomic_and_dimensions_do_not_wrap() {
        let mut output = [123; 8];
        for p in [Predictor::Weighted, Predictor::Best, Predictor::Variable] {
            assert_eq!(
                residual_row(&[0; 8], 8, 8, 0, p, &mut output),
                Err(RowError::UnsupportedPredictor(p))
            );
            assert_eq!(output, [123; 8]);
        }
        for (pixels, w, stride, y, error) in [
            (&[][..], 0, 0, 0, PredictorError::InvalidDimensions),
            (&[][..], 2, 1, 0, PredictorError::InvalidDimensions),
            (&[0][..], 2, 2, 0, PredictorError::BufferTooSmall),
            (&[0; 3][..], 2, 2, 1, PredictorError::BufferTooSmall),
            (&[][..], 1, 2, usize::MAX, PredictorError::DimensionOverflow),
            (&[][..], 1, 1, usize::MAX, PredictorError::DimensionOverflow),
        ] {
            assert_eq!(
                residual_row(pixels, w, stride, y, Predictor::Left, &mut output),
                Err(RowError::Plane(error))
            );
            assert_eq!(output, [123; 8]);
        }
        assert_eq!(
            residual_row(&[0; 9], 9, 9, 0, Predictor::Left, &mut output),
            Err(RowError::OutputTooSmall)
        );
        assert_eq!(output, [123; 8]);
        residual_row(&[-9], 1, usize::MAX, 0, Predictor::Left, &mut output).unwrap();
        assert_eq!(output, [-9, 123, 123, 123, 123, 123, 123, 123]);
    }
}
