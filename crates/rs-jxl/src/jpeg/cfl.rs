// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! JPEG AC chroma-from-luma preparation, not a JPEG-to-JXL encoder.
//!
//! Source: enc_frame.cc::ComputeJPEGTranscodingData / FindAvgIndexOfSumMaximum,
//! chroma_from_luma.{h,cc}, quantizer.h and dec_group.cc at
//! b866b3defec13494bfbaf705db93581cda118e37. Input coefficients stay borrowed.

use std::fmt;

use super::data::BLOCK_SIZE;
use super::transcode_layout::{
    BlockView, CFL_FIXED_POINT_PRECISION, CFL_FIXED_POINT_RATIO_MAX, TranscodeLayout,
};
use crate::execution;

pub const COLOR_TILE_DIM_IN_BLOCKS: u32 = 8;
pub const DEFAULT_COLOR_FACTOR: i32 = 84;
const FIXED_ONE: i32 = 1 << CFL_FIXED_POINT_PRECISION;
const ROUND: i32 = FIXED_ONE / 2;
const OFFSET: usize = 127;

/// Explicit counterpart of force_cfl_jpeg_recompression (true in source defaults).
/// Enabled still requires three components and relative 4:4:4 sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CflMode {
    Disabled,
    Enabled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CflError {
    InvalidChannel(usize),
    BlockOutOfBounds { channel: usize, x: u32, y: u32 },
    QuantizationRatioOutOfRange { coefficient: usize, ratio: i32 },
    ArithmeticOverflow { coefficient: usize },
    RestoredCoefficientOutOfRange { coefficient: usize, value: i64 },
}

impl fmt::Display for CflError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidChannel(channel) => write!(f, "invalid JPEG CfL channel {channel}"),
            Self::BlockOutOfBounds { channel, x, y } => {
                write!(f, "JPEG CfL block ({x}, {y}) outside channel {channel}")
            }
            Self::QuantizationRatioOutOfRange { coefficient, ratio } => write!(
                f,
                "JPEG CfL quantization ratio {ratio} out of range at coefficient {coefficient}"
            ),
            Self::ArithmeticOverflow { coefficient } => write!(
                f,
                "JPEG CfL signed-32-bit arithmetic overflow at coefficient {coefficient}"
            ),
            Self::RestoredCoefficientOutOfRange { coefficient, value } => write!(
                f,
                "restored JPEG coefficient {coefficient} ({value}) outside signed-16-bit range"
            ),
        }
    }
}
impl std::error::Error for CflError {}

/// Source RatioJPEG uses signed division truncating toward zero, not a shift.
pub fn ratio_jpeg(factor: i8) -> i32 {
    i32::from(factor) * FIXED_ONE / DEFAULT_COLOR_FACTOR
}

fn prediction(luma: i16, ratio: i32, factor: i8, coefficient: usize) -> Result<i32, CflError> {
    if !(0..=CFL_FIXED_POINT_RATIO_MAX).contains(&ratio) {
        return Err(CflError::QuantizationRatioOutOfRange { coefficient, ratio });
    }
    let overflow = CflError::ArithmeticOverflow { coefficient };
    let scale = ratio_jpeg(factor)
        .checked_mul(ratio)
        .and_then(|v| v.checked_add(ROUND))
        .ok_or_else(|| overflow.clone())?
        >> CFL_FIXED_POINT_PRECISION;
    Ok(i32::from(luma)
        .checked_mul(scale)
        .and_then(|v| v.checked_add(ROUND))
        .ok_or(overflow)?
        >> CFL_FIXED_POINT_PRECISION)
}

fn transpose(index: usize) -> usize {
    (index % 8) * 8 + index / 8
}

/// Natural-order quantized JPEG input -> transposed JXL AC residuals.
/// `luma` and `ratios` are also in natural order; ratios use 11 fractional bits.
/// DC is copied unchanged. On any error, `output` is unchanged.
///
/// Signed i16 input is never clamped or reinterpreted as unsigned. Combinations
/// exceeding the source's signed-i32 intermediate range are explicitly rejected.
pub fn forward_ac_transposed(
    input: &[i16; BLOCK_SIZE],
    luma: &[i16; BLOCK_SIZE],
    ratios: &[i32; BLOCK_SIZE],
    factor: i8,
    output: &mut [i32; BLOCK_SIZE],
) -> Result<(), CflError> {
    let mut block = [0; BLOCK_SIZE];
    block[0] = i32::from(input[0]);
    for coefficient in 1..BLOCK_SIZE {
        let predicted = prediction(luma[coefficient], ratios[coefficient], factor, coefficient)?;
        // A checked i32 prediction after >>11 cannot overflow this subtraction.
        block[transpose(coefficient)] = i32::from(input[coefficient]) - predicted;
    }
    *output = block;
    Ok(())
}

/// Transposed residuals + original/restored natural-order luma -> natural JPEG.
/// DC comes from `input[0]`, unchanged; a frame decoder supplies it separately.
/// Out-of-i16 reconstruction is an error, never saturation. Output is atomic.
pub fn inverse_ac_transposed(
    input: &[i32; BLOCK_SIZE],
    luma: &[i16; BLOCK_SIZE],
    ratios: &[i32; BLOCK_SIZE],
    factor: i8,
    output: &mut [i16; BLOCK_SIZE],
) -> Result<(), CflError> {
    let mut block = [0; BLOCK_SIZE];
    for (coefficient, restored) in block.iter_mut().enumerate() {
        let predicted = if coefficient == 0 {
            0
        } else {
            prediction(luma[coefficient], ratios[coefficient], factor, coefficient)?
        };
        let value = i64::from(input[transpose(coefficient)]) + i64::from(predicted);
        *restored = i16::try_from(value)
            .map_err(|_| CflError::RestoredCoefficientOutOfRange { coefficient, value })?;
    }
    *output = block;
    Ok(())
}

/// Two immutable row-major maps tied to the exact layout used for fitting.
/// Map 0 predicts JXL channel 0 from channel 1; map 1 predicts channel 2.
#[derive(Debug)]
pub struct JpegCfl<'layout, 'jpeg> {
    layout: &'layout TranscodeLayout<'jpeg>,
    enabled: bool,
    dimensions: (u32, u32),
    maps: [Vec<i8>; 2],
}

impl<'layout, 'jpeg> JpegCfl<'layout, 'jpeg> {
    /// Source scalar fitting, dispatched by tile row through the existing
    /// execution policy. A serial scope stays caller-local; no pool is created.
    /// This does not validate that every later forward block fits i32 arithmetic.
    pub fn compute(layout: &'layout TranscodeLayout<'jpeg>, mode: CflMode) -> Self {
        let (width, height) = layout.dimensions_in_blocks();
        let dimensions = (
            width.div_ceil(COLOR_TILE_DIM_IN_BLOCKS),
            height.div_ceil(COLOR_TILE_DIM_IN_BLOCKS),
        );
        let enabled = mode == CflMode::Enabled && layout.can_use_ac_cfl();
        let maps = [0, 2].map(|channel| {
            if !enabled {
                return vec![0; dimensions.0 as usize * dimensions.1 as usize];
            }
            execution::map_with_state(
                dimensions.1 as usize,
                || [0; 257],
                |differences, ty| {
                    (0..dimensions.0)
                        .map(|tx| fit_tile(layout, channel, tx, ty as u32, differences))
                        .collect::<Vec<_>>()
                },
            )
            .into_iter()
            .flatten()
            .collect()
        });
        Self {
            layout,
            enabled,
            dimensions,
            maps,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn dimensions_in_tiles(&self) -> (u32, u32) {
        self.dimensions
    }

    pub fn ytox_map(&self) -> &[i8] {
        &self.maps[0]
    }

    pub fn ytob_map(&self) -> &[i8] {
        &self.maps[1]
    }

    /// JXL channel and full-frame tile coordinates. Luma's factor is always zero.
    pub fn factor(&self, channel: usize, tx: u32, ty: u32) -> Option<i8> {
        if channel >= 3 || tx >= self.dimensions.0 || ty >= self.dimensions.1 {
            return None;
        }
        if channel == 1 {
            return Some(0);
        }
        Some(self.maps[channel / 2][ty as usize * self.dimensions.0 as usize + tx as usize])
    }

    fn block(&self, channel: usize, x: u32, y: u32) -> Result<BlockView<'jpeg>, CflError> {
        self.layout
            .planes()
            .get(channel)
            .ok_or(CflError::InvalidChannel(channel))?
            .block(x, y)
            .ok_or(CflError::BlockOutOfBounds { channel, x, y })
    }

    /// Coordinates are local to the selected plane (also for subsampled input).
    /// Disabled/ineligible modes only transpose; synthetic gray chroma is zero.
    pub fn forward_block(
        &self,
        channel: usize,
        x: u32,
        y: u32,
        output: &mut [i32; BLOCK_SIZE],
    ) -> Result<(), CflError> {
        let input = self.block(channel, x, y)?;
        if !self.enabled || channel == 1 {
            input.write_transposed(output);
            return Ok(());
        }
        let BlockView::Jpeg(input) = input else {
            unreachable!("eligible three-component input")
        };
        let BlockView::Jpeg(luma) = self.block(1, x, y)? else {
            unreachable!("real luma")
        };
        forward_ac_transposed(
            input,
            luma,
            self.layout.planes()[channel].quantization().cfl_ratios(),
            self.factor(
                channel,
                x / COLOR_TILE_DIM_IN_BLOCKS,
                y / COLOR_TILE_DIM_IN_BLOCKS,
            )
            .expect("validated 4:4:4 geometry"),
            output,
        )
    }

    /// Inverse for encoder-side verification, using this layout's borrowed luma.
    /// A decoder with restored luma can instead call inverse_ac_transposed.
    pub fn inverse_block(
        &self,
        channel: usize,
        x: u32,
        y: u32,
        input: &[i32; BLOCK_SIZE],
        output: &mut [i16; BLOCK_SIZE],
    ) -> Result<(), CflError> {
        self.block(channel, x, y)?;
        let mut luma = &[0; BLOCK_SIZE];
        let mut factor = 0;
        if self.enabled && channel != 1 {
            let BlockView::Jpeg(original_luma) = self.block(1, x, y)? else {
                unreachable!("real luma")
            };
            luma = original_luma;
            factor = self
                .factor(
                    channel,
                    x / COLOR_TILE_DIM_IN_BLOCKS,
                    y / COLOR_TILE_DIM_IN_BLOCKS,
                )
                .expect("validated 4:4:4 geometry");
        }
        inverse_ac_transposed(
            input,
            luma,
            self.layout.planes()[channel].quantization().cfl_ratios(),
            factor,
            output,
        )
    }
}

// This is the midpoint of FIRST and LAST maxima, not necessarily a maximum
// itself when two disjoint plateaus tie. The source deliberately uses this rule.
fn average_maximum(differences: &[i32; 257]) -> (usize, i32) {
    let (mut maximum, mut sum, mut first, mut last) = (0, 0, 0, 0);
    for (index, &difference) in differences[..256].iter().enumerate() {
        sum += difference;
        if sum > maximum {
            maximum = sum;
            first = index;
        }
        if sum == maximum {
            last = index;
        }
    }
    ((first + last).div_ceil(2), maximum)
}

fn choose_factor(differences: &[i32; 257]) -> i8 {
    let (best, best_sum) = average_maximum(differences);
    let offset_sum: i32 = differences[..=OFFSET].iter().sum();
    if best_sum > offset_sum + 1 {
        // Source assigns best-127 to int8_t: endpoint 255 becomes -128 on
        // its two's-complement targets. Do not clamp it to +127.
        (best as i32 - OFFSET as i32) as i8
    } else {
        0
    }
}

fn fit_tile(
    layout: &TranscodeLayout<'_>,
    channel: usize,
    tx: u32,
    ty: u32,
    differences: &mut [i32; 257],
) -> i8 {
    differences.fill(0);
    let (width, height) = layout.dimensions_in_blocks();
    let x0 = tx * COLOR_TILE_DIM_IN_BLOCKS;
    let y0 = ty * COLOR_TILE_DIM_IN_BLOCKS;
    let ratios = layout.planes()[channel].quantization().cfl_ratios();
    let scale = DEFAULT_COLOR_FACTOR as f32;
    let threshold = scale * 0.5f32 * 0.9999f32; // quantizer.h::kZeroBiasDefault
    for y in y0..height.min(y0 + COLOR_TILE_DIM_IN_BLOCKS) {
        for x in x0..width.min(x0 + COLOR_TILE_DIM_IN_BLOCKS) {
            let Some(BlockView::Jpeg(luma)) = layout.planes()[1].block(x, y) else {
                unreachable!("validated eligible luma geometry")
            };
            let Some(BlockView::Jpeg(side)) = layout.planes()[channel].block(x, y) else {
                unreachable!("validated eligible side geometry")
            };
            for coefficient in 1..BLOCK_SIZE {
                // Preserve source f32 operations and order, without FMA. Both
                // base correlations are zero in ColorCorrelationMap::Create(false).
                let scaled_m = f32::from(luma[coefficient])
                    * (1.0f32 / FIXED_ONE as f32)
                    * ratios[coefficient] as f32;
                let scaled_s = scale * f32::from(side[coefficient]) + OFFSET as f32 * scaled_m;
                if scaled_m.abs() > 1e-8f32 {
                    let (mut from, mut to) = if scaled_m > 0.0 {
                        (
                            (scaled_s - threshold) / scaled_m,
                            (scaled_s + threshold) / scaled_m,
                        )
                    } else {
                        (
                            (scaled_s + threshold) / scaled_m,
                            (scaled_s - threshold) / scaled_m,
                        )
                    };
                    // Clamping both ends would invent votes outside [0,255].
                    if from < 0.0 {
                        from = 0.0;
                    }
                    if to > 255.0 {
                        to = 255.0;
                    }
                    if from <= to {
                        differences[from.ceil() as usize] += 1;
                        differences[(to + 1.0).floor() as usize] -= 1;
                    }
                }
            }
        }
    }
    choose_factor(differences)
}

#[cfg(test)]
mod tests {
    use super::super::data::{Component, ComponentScanInfo, JpegData, QuantTable, ScanInfo};
    use super::*;

    fn fixture(width: u32, height: u32, sampling: &[(u32, u32)]) -> JpegData {
        let max_h = sampling.iter().map(|s| s.0).max().unwrap();
        let max_v = sampling.iter().map(|s| s.1).max().unwrap();
        JpegData {
            width,
            height,
            precision: 8,
            components: sampling
                .iter()
                .enumerate()
                .map(|(c, &(h, v))| {
                    let width_in_blocks = width.div_ceil(8 * max_h) * h;
                    let height_in_blocks = height.div_ceil(8 * max_v) * v;
                    Component {
                        id: c as u32 + 1,
                        h_samp_factor: h,
                        v_samp_factor: v,
                        quant_index: c as u32,
                        width_in_blocks,
                        height_in_blocks,
                        coefficients: vec![
                            0;
                            width_in_blocks as usize
                                * height_in_blocks as usize
                                * BLOCK_SIZE
                        ],
                    }
                })
                .collect(),
            quant: (0..sampling.len())
                .map(|c| QuantTable {
                    index: c as u32,
                    values: [1; BLOCK_SIZE],
                    ..QuantTable::default()
                })
                .collect(),
            scan_info: vec![ScanInfo {
                components: (0..sampling.len())
                    .map(|component_index| ComponentScanInfo {
                        component_index,
                        ..ComponentScanInfo::default()
                    })
                    .collect(),
                se: 63,
                quant_table_indices: [Some(0), Some(1), Some(2), None],
                ..ScanInfo::default()
            }],
            ..JpegData::default()
        }
    }

    fn fitted<'layout, 'jpeg>(layout: &'layout TranscodeLayout<'jpeg>) -> JpegCfl<'layout, 'jpeg> {
        execution::serial(|| JpegCfl::compute(layout, CflMode::Enabled))
    }

    // Independent brute-force bin scores: no difference array or prefix scan.
    // The source's floor(to+1) is intentional, including its f32 rounding.
    fn brute_factor(layout: &TranscodeLayout<'_>, channel: usize, tx: u32, ty: u32) -> i8 {
        let mut scores = [0i32; 256];
        let (width, height) = layout.dimensions_in_blocks();
        let ratios = layout.planes()[channel].quantization().cfl_ratios();
        for y in ty * 8..height.min((ty + 1) * 8) {
            for x in tx * 8..width.min((tx + 1) * 8) {
                let BlockView::Jpeg(luma) = layout.planes()[1].block(x, y).unwrap() else {
                    panic!()
                };
                let BlockView::Jpeg(side) = layout.planes()[channel].block(x, y).unwrap() else {
                    panic!()
                };
                for i in 1..64 {
                    let m = (f32::from(luma[i]) * (1.0f32 / 2048.0)) * ratios[i] as f32;
                    if m.abs() <= 1e-8f32 {
                        continue;
                    }
                    let s = 84.0f32 * f32::from(side[i]) + 127.0f32 * m;
                    let radius = (84.0f32 * 0.5) * 0.9999f32;
                    let a = (s - radius) / m;
                    let b = (s + radius) / m;
                    let start = a.min(b).ceil();
                    let end = (a.max(b) + 1.0).floor();
                    for (bin, score) in scores.iter_mut().enumerate() {
                        if bin as f32 >= start && (bin as f32) < end {
                            *score += 1;
                        }
                    }
                }
            }
        }
        let maximum = *scores.iter().max().unwrap();
        if maximum <= scores[127] + 1 {
            return 0;
        }
        let first = scores.iter().position(|&s| s == maximum).unwrap();
        let last = scores.iter().rposition(|&s| s == maximum).unwrap();
        ((first + last).div_ceil(2) as i32 - 127) as i8
    }

    #[test]
    fn ratio_jpeg_golden_signed_truncation_and_rounding() {
        for (factor, expected) in [
            (-128, -3120),
            (-127, -3096),
            (-1, -24),
            (0, 0),
            (1, 24),
            (42, 1024),
            (84, 2048),
            (127, 3096),
        ] {
            assert_eq!(ratio_jpeg(factor), expected);
        }
        // Rounding is +1024 then arithmetic >>11, including negative halves.
        for (luma, factor, expected) in [
            (1, 42, 1),
            (-1, 42, 0),
            (1, -42, 0),
            (-1, -42, 1),
            (3, -42, -1),
        ] {
            assert_eq!(prediction(luma, 2048, factor, 1).unwrap(), expected);
        }
    }

    #[test]
    fn source_maximum_ties_empty_votes_and_improvement_gate() {
        let mut differences = [0; 257];
        assert_eq!(average_maximum(&differences), (128, 0));
        assert_eq!(choose_factor(&differences), 0);
        differences[10] = 6;
        differences[11] = -6;
        differences[20] = 6;
        differences[21] = -6;
        assert_eq!(average_maximum(&differences), (15, 6));
        assert_eq!(choose_factor(&differences), -112);
        differences.fill(0);
        differences[169] = 1;
        differences[170] = -1;
        assert_eq!(choose_factor(&differences), 0);
        differences[169] = 2;
        differences[170] = -2;
        assert_eq!(choose_factor(&differences), 42);
    }

    #[test]
    fn fitted_golden_maps_keep_dc_and_reconstruct_negative_coefficients() {
        let mut jpeg = fixture(8, 8, &[(1, 1); 3]);
        for i in 1..64 {
            let sign = if i % 2 == 0 { -1 } else { 1 };
            jpeg.components[0].coefficients[i] = 84 * sign;
            jpeg.components[1].coefficients[i] = 42 * sign;
            jpeg.components[2].coefficients[i] = -21 * sign;
        }
        for (c, dc) in jpeg.components.iter_mut().zip([i16::MIN, i16::MAX, -1234]) {
            c.coefficients[0] = dc;
        }
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let cfl = fitted(&layout);
        assert!(cfl.is_enabled());
        assert_eq!(cfl.ytox_map(), &[42]);
        assert_eq!(cfl.ytob_map(), &[-21]);
        for channel in 0..3 {
            let mut residual = [0; 64];
            cfl.forward_block(channel, 0, 0, &mut residual).unwrap();
            let source = &jpeg.components[layout.jpeg_order()[channel]].coefficients;
            assert_eq!(residual[0], i32::from(source[0]));
            if channel != 1 {
                assert_eq!(residual[1..], [0; 63]);
            }
            let mut restored = [0; 64];
            cfl.inverse_block(channel, 0, 0, &residual, &mut restored)
                .unwrap();
            assert_eq!(restored.as_slice(), source);
        }
    }

    #[test]
    fn signed_byte_upper_endpoint_is_not_clamped() {
        let mut jpeg = fixture(8, 8, &[(1, 1); 3]);
        jpeg.components[0].coefficients[1..].fill(84);
        jpeg.components[1].coefficients[1..].fill(128);
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let cfl = fitted(&layout);
        assert_eq!(cfl.ytox_map(), &[-128]);
        assert_eq!(brute_factor(&layout, 0, 0, 0), -128);
        let mut residual = [0; 64];
        cfl.forward_block(0, 0, 0, &mut residual).unwrap();
        let mut restored = [0; 64];
        cfl.inverse_block(0, 0, 0, &residual, &mut restored)
            .unwrap();
        assert_eq!(&restored[1..], &[128; 63]);
    }

    #[test]
    fn tile_fit_matches_brute_bins_with_partial_tiles_and_quantization_limits() {
        let mut jpeg = fixture(73, 65, &[(1, 1); 3]);
        for table in &mut jpeg.quant {
            table.precision = 1;
        }
        for i in 1..64 {
            // Include zero, near-maximum, nonintegral and unity fixed-point ratios.
            let (y, x, b) = [(1, 65535, 1), (65535, 256, 257), (3, 7, 5), (1, 1, 1)][i % 4];
            jpeg.quant[0].values[i] = y;
            jpeg.quant[1].values[i] = x;
            jpeg.quant[2].values[i] = b;
        }
        let mut state = 1u32;
        for block in 0..jpeg.components[0].coefficients.len() / 64 {
            for i in 1..64 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let luma = ((state >> 16) as i16) / 64;
                let factor = if block % 10 < 8 { 42i32 } else { -21i32 };
                jpeg.components[0].coefficients[block * 64 + i] = luma;
                jpeg.components[1].coefficients[block * 64 + i] =
                    (i32::from(luma) * factor / 84) as i16;
                jpeg.components[2].coefficients[block * 64 + i] = -luma / 4;
            }
        }
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let cfl = fitted(&layout);
        assert_eq!(cfl.dimensions_in_tiles(), (2, 2));
        assert_eq!(cfl.ytox_map(), &[98, -49, 98, -49]);
        assert_eq!(cfl.ytob_map(), &[-21, -21, -21, -21]);
        for channel in [0, 2] {
            for ty in 0..2 {
                for tx in 0..2 {
                    assert_eq!(
                        cfl.factor(channel, tx, ty),
                        Some(brute_factor(&layout, channel, tx, ty))
                    );
                }
            }
        }
    }

    #[test]
    fn zero_luma_and_outside_candidate_ranges_cast_no_votes() {
        let mut jpeg = fixture(8, 8, &[(1, 1); 3]);
        jpeg.components[1].coefficients.fill(i16::MAX);
        jpeg.components[2].coefficients.fill(i16::MIN);
        for luma in [0, 1, -1] {
            jpeg.components[0].coefficients.fill(luma);
            let layout = TranscodeLayout::new(&jpeg).unwrap();
            let cfl = fitted(&layout);
            assert_eq!(cfl.ytox_map(), &[0]);
            assert_eq!(cfl.ytob_map(), &[0]);
            assert_eq!(brute_factor(&layout, 0, 0, 0), 0);
            assert_eq!(brute_factor(&layout, 2, 0, 0), 0);
        }
    }

    #[test]
    fn explicit_mode_gray_subsampling_and_rgb_guards() {
        for sampling in [&[(1, 1); 3][..], &[(2, 2), (1, 1), (1, 1)], &[(1, 1)]] {
            let mut jpeg = fixture(17, 9, sampling);
            for c in &mut jpeg.components {
                for (i, value) in c.coefficients.iter_mut().enumerate() {
                    *value = i as i16 - 123;
                }
            }
            let layout = TranscodeLayout::new(&jpeg).unwrap();
            for mode in [CflMode::Disabled, CflMode::Enabled] {
                let cfl = execution::serial(|| JpegCfl::compute(&layout, mode));
                let expected_enabled = sampling == [(1, 1); 3] && mode == CflMode::Enabled;
                assert_eq!(cfl.is_enabled(), expected_enabled);
                if !expected_enabled {
                    assert!(cfl.ytox_map().iter().chain(cfl.ytob_map()).all(|&f| f == 0));
                    for channel in 0..3 {
                        let plane = &layout.planes()[channel];
                        let (w, h) = plane.dimensions_in_blocks();
                        let mut expected = [0; 64];
                        plane
                            .block(w - 1, h - 1)
                            .unwrap()
                            .write_transposed(&mut expected);
                        let mut actual = [0; 64];
                        cfl.forward_block(channel, w - 1, h - 1, &mut actual)
                            .unwrap();
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
        let mut jpeg = fixture(8, 8, &[(1, 1); 3]);
        for (component, &id) in jpeg.components.iter_mut().zip(b"RGB") {
            component.id = u32::from(id);
        }
        jpeg.components[1].coefficients.fill(84);
        jpeg.components[0].coefficients.fill(-42);
        jpeg.components[2].coefficients.fill(21);
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let cfl = fitted(&layout);
        assert_eq!(layout.jpeg_order(), [0, 1, 2]);
        assert_eq!(cfl.ytox_map(), &[-42]);
        assert_eq!(cfl.ytob_map(), &[21]);
    }

    #[test]
    fn full_i16_domain_round_trips_with_independent_wide_integer_oracle() {
        let input = std::array::from_fn(|i| [i16::MIN, -1025, -1, 0, 1, 1024, i16::MAX][i % 7]);
        let mut luma = [0; 64];
        for factor in [i8::MIN, -127, -84, -1, 0, 1, 42, 84, 127] {
            for ratio in [0, 1, 2048, CFL_FIXED_POINT_RATIO_MAX] {
                for (i, coefficient) in luma.iter_mut().enumerate() {
                    *coefficient = if ratio > 2048 {
                        if i % 2 == 0 { -1024 } else { 1024 }
                    } else {
                        input[i]
                    };
                }
                let mut ratios = [ratio; 64];
                ratios[0] = -1; // DC must neither validate this nor use it.
                let mut residual = [0; 64];
                forward_ac_transposed(&input, &luma, &ratios, factor, &mut residual).unwrap();
                assert_eq!(residual[0], i32::from(input[0]));
                for i in 1..64 {
                    let scale =
                        (i64::from(factor) * 2048 / 84 * i64::from(ratio) + 1024).div_euclid(2048);
                    let predicted = (i64::from(luma[i]) * scale + 1024).div_euclid(2048);
                    assert_eq!(
                        i64::from(residual[(i % 8) * 8 + i / 8]),
                        i64::from(input[i]) - predicted
                    );
                }
                let mut restored = [0; 64];
                inverse_ac_transposed(&residual, &luma, &ratios, factor, &mut restored).unwrap();
                assert_eq!(restored, input);
            }
        }
    }

    #[test]
    fn arithmetic_and_range_errors_leave_output_unchanged() {
        let luma = [i16::MAX; 64];
        let input = [i16::MIN; 64];
        let mut ratios = [CFL_FIXED_POINT_RATIO_MAX; 64];
        let mut output = [9876; 64];
        assert_eq!(
            forward_ac_transposed(&input, &luma, &ratios, 127, &mut output),
            Err(CflError::ArithmeticOverflow { coefficient: 1 })
        );
        assert_eq!(output, [9876; 64]);
        for bad in [-1, CFL_FIXED_POINT_RATIO_MAX + 1] {
            ratios.fill(2048);
            ratios[63] = bad;
            assert_eq!(
                forward_ac_transposed(&input, &luma, &ratios, 0, &mut output),
                Err(CflError::QuantizationRatioOutOfRange {
                    coefficient: 63,
                    ratio: bad
                })
            );
            assert_eq!(output, [9876; 64]);
        }
        ratios.fill(2048);
        let mut residual = [0; 64];
        residual[63] = i32::MAX;
        let mut restored = [123; 64];
        assert!(matches!(
            inverse_ac_transposed(&residual, &luma, &ratios, 0, &mut restored),
            Err(CflError::RestoredCoefficientOutOfRange {
                coefficient: 63,
                ..
            })
        ));
        assert_eq!(restored, [123; 64]);
        ratios.fill(CFL_FIXED_POINT_RATIO_MAX);
        assert_eq!(
            inverse_ac_transposed(&residual, &luma, &ratios, -128, &mut restored),
            Err(CflError::ArithmeticOverflow { coefficient: 1 })
        );
        assert_eq!(restored, [123; 64]);
    }

    #[test]
    fn checked_coordinates_and_layout_bound_application() {
        let jpeg = fixture(8, 8, &[(1, 1); 3]);
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let cfl = fitted(&layout);
        assert_eq!(cfl.factor(1, 0, 0), Some(0));
        assert_eq!(cfl.factor(3, 0, 0), None);
        assert_eq!(cfl.factor(0, 1, 0), None);
        assert_eq!(cfl.factor(0, 0, 1), None);
        let mut output = [77; 64];
        assert_eq!(
            cfl.forward_block(3, 0, 0, &mut output),
            Err(CflError::InvalidChannel(3))
        );
        assert_eq!(
            cfl.forward_block(0, 1, 0, &mut output),
            Err(CflError::BlockOutOfBounds {
                channel: 0,
                x: 1,
                y: 0
            })
        );
        assert_eq!(output, [77; 64]);
        let mut restored = [55; 64];
        assert_eq!(
            cfl.inverse_block(0, 0, 1, &output, &mut restored),
            Err(CflError::BlockOutOfBounds {
                channel: 0,
                x: 0,
                y: 1
            })
        );
        assert_eq!(restored, [55; 64]);
    }

    #[test]
    fn tile_row_dispatch_matches_serial_in_caller_owned_pools() {
        let mut jpeg = fixture(73, 129, &[(1, 1); 3]);
        for i in 0..jpeg.components[0].coefficients.len() {
            let y = if i % 3 == 0 { -84 } else { 84 };
            jpeg.components[0].coefficients[i] = y;
            jpeg.components[1].coefficients[i] = if i / 640 < 8 { y / 2 } else { -y / 4 };
            jpeg.components[2].coefficients[i] = -y / 2;
        }
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let expected = fitted(&layout);
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let actual =
                execution::with_pool(&pool, || JpegCfl::compute(&layout, CflMode::Enabled));
            assert_eq!(actual.ytox_map(), expected.ytox_map());
            assert_eq!(actual.ytob_map(), expected.ytob_map());
        }
    }
}
