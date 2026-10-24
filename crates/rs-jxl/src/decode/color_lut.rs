//! Direct evaluation of parsed ICC mft1/mft2 CMYK tables. Keeping input curves,
//! CLUT interpolation, output curves and PCS conversion in that order avoids
//! resampling a nonlinear Lab-to-XYZ transform into a second coarse grid.
//! ICC byte parsing remains the responsibility of moxcms.
//!
//! Compatibility reference: libjxl at b866b3defec13494bfbaf705db93581cda118e37,
//! lib/jxl/cms/jxl_cms.cc and its local third_party/skcms/{skcms.cc,
//! src/skcms_public.h,src/Transform_inl.h}. The default table priority and
//! lab_to_xyz scalar operation follow those sources, not ICC legacy scaling.

// Translated skcms portions: Copyright (c) 2018 Google Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

use moxcms::{
    ColorProfile, DataColorSpace, LutDataType, LutStore, LutType, LutWarehouse, ProfileClass,
    ProfileVersion, RenderingIntent, Xyzd,
};

use super::color::invalid;
use crate::Result;

pub(super) struct CmykLut {
    lut: LutDataType,
    lab: bool,
    lab_scale: f32,
    icc_conventions: bool,
}

impl CmykLut {
    pub(super) fn new(
        source: &ColorProfile,
        destination: &ColorProfile,
        icc_conventions: bool,
    ) -> Result<Option<Self>> {
        if source.color_space != DataColorSpace::Cmyk {
            return Ok(None);
        }
        let selected = if icc_conventions {
            let selected = match destination.rendering_intent {
                RenderingIntent::Perceptual => &source.lut_a_to_b_perceptual,
                RenderingIntent::Saturation => &source.lut_a_to_b_saturation,
                _ => &source.lut_a_to_b_colorimetric,
            };
            selected.as_ref().or(source.lut_a_to_b_perceptual.as_ref())
        } else {
            // skcms_Parse uses {0, 1} regardless of either profile's intent.
            // A2B2 alone must not silently become an alternate compatibility LUT.
            Some(
                source
                    .lut_a_to_b_perceptual
                    .as_ref()
                    .or(source.lut_a_to_b_colorimetric.as_ref())
                    .ok_or_else(|| invalid("djxl CMYK conversion requires A2B0 or A2B1"))?,
            )
        };
        let Some(LutWarehouse::Lut(lut)) = selected else {
            return Ok(None);
        };
        if lut.num_input_channels != 4
            || lut.num_output_channels != 3
            || !matches!(lut.lut_type, LutType::Lut8 | LutType::Lut16)
            || !matches!(source.pcs, DataColorSpace::Lab | DataColorSpace::Xyz)
        {
            return Err(invalid("unsupported CMYK ICC table layout or PCS"));
        }
        let n = usize::from(lut.num_clut_grid_points);
        let input_entries = usize::from(lut.num_input_table_entries);
        let output_entries = usize::from(lut.num_output_table_entries);
        if n < 2
            || input_entries < 2
            || output_entries < 2
            || input_entries > 4096
            || output_entries > 4096
            || length(&lut.input_table) != input_entries * 4
            || length(&lut.output_table) != output_entries * 3
            || n.checked_pow(4).and_then(|v| v.checked_mul(3)) != Some(length(&lut.clut_table))
        {
            return Err(invalid("invalid CMYK ICC table dimensions"));
        }
        Ok(Some(Self {
            lut: lut.clone(),
            lab: source.pcs == DataColorSpace::Lab,
            // Legacy 16-bit Lab tables encode L*=100 at 0xff00, not 0xffff.
            lab_scale: if icc_conventions && lut.lut_type == LutType::Lut16 {
                65535.0 / 65280.0
            } else {
                1.0
            },
            icc_conventions,
        }))
    }

    pub(super) fn transform(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if !input.len().is_multiple_of(4)
            || output.len() / 3 != input.len() / 4
            || !output.len().is_multiple_of(3)
        {
            return Err(invalid("CMYK ICC buffer dimensions"));
        }
        for (input, output) in input.chunks_exact(4).zip(output.chunks_exact_mut(3)) {
            if input
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            {
                return Err(invalid("CMYK ICC input must be finite and in [0, 1]"));
            }
            let grid = usize::from(self.lut.num_clut_grid_points);
            let mut low = [0usize; 4];
            let mut high = [0usize; 4];
            let mut fraction = [0.0; 4];
            for c in 0..4 {
                let v = curve(
                    &self.lut.input_table,
                    usize::from(self.lut.num_input_table_entries),
                    c,
                    input[c],
                    self.icc_conventions,
                );
                let position = v * (grid - 1) as f32;
                (low[c], high[c]) = table_indices(position, grid, self.icc_conventions);
                fraction[c] = position - low[c] as f32;
            }
            let mut pcs = [0.0; 3];
            for corner in 0..16 {
                let mut index = 0;
                let mut weights = [0.0; 4];
                for c in 0..4 {
                    let shift = if self.icc_conventions { 3 - c } else { c };
                    let upper = (corner >> shift) & 1 != 0;
                    index = index * grid + if upper { high[c] } else { low[c] };
                    weights[c] = if !upper {
                        1.0 - fraction[c]
                    } else {
                        fraction[c]
                    };
                }
                // skcms visits C in the low corner bit and multiplies weights
                // in C,K,Y,M order. Reassociation changes cancellation near
                // black after Lab->XYZ->RGB, even though the LUT is identical.
                let weight = if self.icc_conventions {
                    weights[0] * weights[1] * weights[2] * weights[3]
                } else {
                    weights[0] * weights[3] * weights[2] * weights[1]
                };
                for (c, v) in pcs.iter_mut().enumerate() {
                    *v += weight * value(&self.lut.clut_table, index * 3 + c, self.icc_conventions);
                }
            }
            for (c, v) in pcs.iter_mut().enumerate() {
                *v = curve(
                    &self.lut.output_table,
                    usize::from(self.lut.num_output_table_entries),
                    c,
                    *v,
                    self.icc_conventions,
                );
            }
            let xyz = if self.lab && !self.icc_conventions {
                skcms_lab_to_xyz(pcs)
            } else if self.lab {
                let l = pcs[0] * self.lab_scale * 100.0;
                let a = pcs[1] * self.lab_scale * 255.0 - 128.0;
                let b = pcs[2] * self.lab_scale * 255.0 - 128.0;
                let fy = (l + 16.0) / 116.0;
                let inverse = |v: f32| {
                    if v > 6.0 / 29.0 {
                        v * v * v
                    } else {
                        (v - 4.0 / 29.0) * (108.0 / 841.0)
                    }
                };
                [
                    0.9642 * inverse(fy + a / 500.0),
                    inverse(fy),
                    0.8249 * inverse(fy - b / 200.0),
                ]
            } else if self.icc_conventions {
                pcs.map(|v| v * (65535.0 / 32768.0))
            } else {
                // read_tag_mft1/mft2 leaves normalized XYZ tables unscaled.
                pcs
            };
            output.copy_from_slice(&xyz);
        }
        Ok(())
    }
}

fn skcms_lab_to_xyz(pcs: [f32; 3]) -> [f32; 3] {
    let l = pcs[0] * 100.0;
    let a = pcs[1] * 255.0 - 128.0;
    let b = pcs[2] * 255.0 - 128.0;
    let y = (l + 16.0) * (1.0 / 116.0);
    let x = y + a * (1.0 / 500.0);
    let z = y - b * (1.0 / 200.0);
    let inverse = |v: f32| {
        let cube = v * v * v;
        if cube > 0.008856 {
            cube
        } else {
            (v - 16.0 / 116.0) * (1.0 / 7.787)
        }
    };
    [inverse(x) * 0.9642, inverse(y), inverse(z) * 0.8249]
}

pub(super) fn absolute_scale(
    source: &ColorProfile,
    destination: &ColorProfile,
) -> Result<[f64; 3]> {
    if destination.rendering_intent != RenderingIntent::AbsoluteColorimetric {
        return Ok([1.0; 3]);
    }
    // ICC absolute intent, full chromatic adaptation: preserve media-white Y.
    // Missing media-white tags and legacy display profiles use PCS D50.
    let media = |p: &ColorProfile| {
        if p.profile_class == ProfileClass::DisplayDevice && p.version() < ProfileVersion::V4_0 {
            Xyzd::new(0.9642, 1.0, 0.8249)
        } else {
            p.media_white_point
                .unwrap_or(Xyzd::new(0.9642, 1.0, 0.8249))
        }
    };
    let src = media(source);
    let dst = media(destination);
    let mut scale = [1.0; 3];
    for (out, (a, b)) in scale
        .iter_mut()
        .zip([src.x, src.y, src.z].into_iter().zip([dst.x, dst.y, dst.z]))
    {
        if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
            return Err(invalid("invalid ICC media white point"));
        }
        *out = a / b;
    }
    Ok(scale)
}

fn length(table: &LutStore) -> usize {
    match table {
        LutStore::Store8(v) => v.len(),
        LutStore::Store16(v) => v.len(),
    }
}
fn value(table: &LutStore, index: usize, icc_conventions: bool) -> f32 {
    let (value, max) = match table {
        LutStore::Store8(v) => (f32::from(v[index]), 255.0),
        LutStore::Store16(v) => (f32::from(v[index]), 65535.0),
    };
    if icc_conventions {
        value / max
    } else {
        // skcms normalizes each sample with a rounded F32 reciprocal.
        value * (1.0 / max)
    }
}

fn table_indices(position: f32, entries: usize, icc_conventions: bool) -> (usize, usize) {
    if icc_conventions {
        let low = (position as usize).min(entries - 2);
        (low, low + 1)
    } else {
        // At exact nodes skcms samples the same entry twice, including the
        // final node. Using the preceding segment at t=1 can round differently.
        let low = (position as usize).min(entries - 1);
        let high = f32::from_bits((position + 1.0).to_bits() - 1) as usize;
        (low, high.min(entries - 1))
    }
}

fn curve(
    table: &LutStore,
    entries: usize,
    channel: usize,
    input: f32,
    icc_conventions: bool,
) -> f32 {
    let position = input.clamp(0.0, 1.0) * (entries - 1) as f32;
    let (low, high) = table_indices(position, entries, icc_conventions);
    let a = value(table, channel * entries + low, icc_conventions);
    let b = value(table, channel * entries + high, icc_conventions);
    a + (b - a) * (position - low as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxcms::Matrix3d;

    fn profile(lab: bool, samples: [u16; 3]) -> ColorProfile {
        let mut p = ColorProfile::new_srgb();
        p.profile_class = ProfileClass::InputDevice;
        p.color_space = DataColorSpace::Cmyk;
        p.pcs = if lab {
            DataColorSpace::Lab
        } else {
            DataColorSpace::Xyz
        };
        p.lut_a_to_b_perceptual = Some(LutWarehouse::Lut(LutDataType {
            num_input_channels: 4,
            num_output_channels: 3,
            num_clut_grid_points: 2,
            matrix: Matrix3d::IDENTITY,
            num_input_table_entries: 2,
            num_output_table_entries: 2,
            input_table: LutStore::Store16([0, 65535].repeat(4)),
            output_table: LutStore::Store16([0, 65535].repeat(3)),
            clut_table: LutStore::Store16(samples.repeat(16)),
            lut_type: LutType::Lut16,
        }));
        p
    }

    #[test]
    fn legacy_lab_white_and_black_are_neutral_without_resampling() {
        let destination = ColorProfile::new_srgb();
        for (l, expected) in [(65280, [0.9642, 1.0, 0.8249]), (0, [0.0; 3])] {
            let lut = CmykLut::new(&profile(true, [l, 32768, 32768]), &destination, true)
                .unwrap()
                .unwrap();
            let mut output = [0.0; 6];
            lut.transform(&[0.0; 8], &mut output).unwrap();
            for chunk in output.chunks_exact(3) {
                for (actual, expected) in chunk.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn xyz_above_one_and_all_grid_boundaries_are_preserved() {
        let lut = CmykLut::new(
            &profile(false, [49152, 32768, 16384]),
            &ColorProfile::new_srgb(),
            true,
        )
        .unwrap()
        .unwrap();
        for corner in 0..16 {
            let input = std::array::from_fn::<_, 4, _>(|c| ((corner >> c) & 1) as f32);
            let mut out = [0.0; 3];
            lut.transform(&input, &mut out).unwrap();
            assert_eq!(out, [1.5, 1.0, 0.5]);
        }
    }

    #[test]
    fn input_curve_precedes_clut_and_output_curve() {
        let mut p = profile(false, [0; 3]);
        let Some(LutWarehouse::Lut(lut)) = &mut p.lut_a_to_b_perceptual else {
            unreachable!()
        };
        lut.num_input_table_entries = 3;
        lut.input_table = LutStore::Store16([0, 16384, 65535].repeat(4));
        let mut clut = Vec::new();
        for index in 0..16 {
            clut.extend([if index & 8 != 0 { 32768 } else { 0 }; 3]);
        }
        lut.clut_table = LutStore::Store16(clut);
        lut.output_table = LutStore::Store16([0, 32768].repeat(3));
        let lut = CmykLut::new(&p, &ColorProfile::new_srgb(), true)
            .unwrap()
            .unwrap();
        let mut out = [0.0; 3];
        lut.transform(&[0.5; 4], &mut out).unwrap();
        let expected = (16384.0 / 65535.0) * (32768.0 / 65535.0);
        assert!(out.iter().all(|v| (v - expected).abs() < 1e-6));
    }

    #[test]
    fn malformed_buffers_and_tables_are_errors() {
        let mut p = profile(false, [0; 3]);
        let lut = CmykLut::new(&p, &ColorProfile::new_srgb(), true)
            .unwrap()
            .unwrap();
        assert!(lut.transform(&[0.0; 3], &mut [0.0; 3]).is_err());
        assert!(lut.transform(&[f32::NAN; 4], &mut [0.0; 3]).is_err());
        assert!(lut.transform(&[-0.1; 4], &mut [0.0; 3]).is_err());
        let Some(LutWarehouse::Lut(table)) = &mut p.lut_a_to_b_perceptual else {
            unreachable!()
        };
        table.num_clut_grid_points = 255;
        assert!(CmykLut::new(&p, &ColorProfile::new_srgb(), true).is_err());
    }

    fn evaluate(source: &ColorProfile, destination: &ColorProfile, icc: bool) -> [f32; 3] {
        let lut = CmykLut::new(source, destination, icc).unwrap().unwrap();
        let mut output = [0.0; 3];
        lut.transform(&[0.0; 4], &mut output).unwrap();
        output
    }

    #[test]
    fn djxl_uses_a2b0_then_a2b1_but_never_a2b2() {
        let mut source = profile(false, [16384; 3]);
        source.lut_a_to_b_colorimetric = profile(false, [32768; 3]).lut_a_to_b_perceptual;
        source.lut_a_to_b_saturation = profile(false, [49152; 3]).lut_a_to_b_perceptual;
        let mut destination = ColorProfile::new_srgb();
        for intent in [
            RenderingIntent::Perceptual,
            RenderingIntent::RelativeColorimetric,
            RenderingIntent::Saturation,
            RenderingIntent::AbsoluteColorimetric,
        ] {
            source.rendering_intent = intent;
            destination.rendering_intent = intent;
            assert_eq!(
                evaluate(&source, &destination, false),
                [16384.0 / 65535.0; 3]
            );
        }
        source.lut_a_to_b_perceptual = None;
        destination.rendering_intent = RenderingIntent::Perceptual;
        assert_eq!(
            evaluate(&source, &destination, false),
            [32768.0 / 65535.0; 3]
        );
        source.lut_a_to_b_colorimetric = None;
        assert!(CmykLut::new(&source, &destination, false).is_err());
        destination.rendering_intent = RenderingIntent::Saturation;
        assert_eq!(evaluate(&source, &destination, true), [1.5; 3]);
    }

    #[test]
    fn malformed_preferred_a2b0_does_not_fall_back_to_a2b1() {
        let mut source = profile(false, [0; 3]);
        source.lut_a_to_b_colorimetric = source.lut_a_to_b_perceptual.clone();
        let Some(LutWarehouse::Lut(lut)) = &mut source.lut_a_to_b_perceptual else {
            unreachable!()
        };
        lut.num_clut_grid_points = 1;
        assert!(CmykLut::new(&source, &ColorProfile::new_srgb(), false).is_err());
    }

    #[test]
    fn djxl_legacy_lab_keeps_ffff_normalization() {
        // With u16 normalization, legacy neutral a/b=0x8000 becomes -128/257,
        // and L=0xff00 becomes 100*256/257. These are intentionally not D50
        // white or neutral black, unlike the explicit ICC-conventions mode.
        let destination = ColorProfile::new_srgb();
        for l in [0, 5242, 5243, 65280, 65535] {
            let source = profile(true, [l, 32768, 32768]);
            let actual = evaluate(&source, &destination, false);
            let y = (f64::from(l) * 100.0 / 65535.0 + 16.0) / 116.0;
            let inverse = |v: f64| {
                if v.powi(3) > 0.008856 {
                    v.powi(3)
                } else {
                    (v - 16.0 / 116.0) / 7.787
                }
            };
            let expected = [
                0.9642 * inverse(y - (128.0 / 257.0) / 500.0),
                inverse(y),
                0.8249 * inverse(y + (128.0 / 257.0) / 200.0),
            ];
            for (value, expected) in actual.into_iter().zip(expected) {
                assert!((f64::from(value) - expected).abs() < 5e-7);
            }
            if l == 0 {
                assert!(actual[0] < 0.0 && actual[2] > 0.0);
            }
            if l == 65280 {
                assert!(actual[1] < 0.991);
            }
        }
    }

    #[test]
    fn djxl_mft1_lab_and_xyz_use_unit_normalization() {
        let mut source = profile(true, [0; 3]);
        let Some(LutWarehouse::Lut(lut)) = &mut source.lut_a_to_b_perceptual else {
            unreachable!()
        };
        lut.lut_type = LutType::Lut8;
        lut.num_input_table_entries = 256;
        lut.num_output_table_entries = 256;
        let identity: Vec<u8> = (0..=255).collect();
        lut.input_table = LutStore::Store8(identity.repeat(4));
        lut.output_table = LutStore::Store8(identity.repeat(3));
        lut.clut_table = LutStore::Store8([255, 128, 128].repeat(16));
        let destination = ColorProfile::new_srgb();
        let actual = evaluate(&source, &destination, false);
        for (actual, expected) in actual.into_iter().zip([0.9642, 1.0, 0.8249]) {
            assert!((actual - expected).abs() < 5e-7);
        }
        source.pcs = DataColorSpace::Xyz;
        assert_eq!(
            evaluate(&source, &destination, false),
            [1.0, 128.0 / 255.0, 128.0 / 255.0]
        );
    }

    #[test]
    fn djxl_lab_extended_pcs_is_not_clipped() {
        let destination = ColorProfile::new_srgb();
        let black = evaluate(&profile(true, [0, 0, 65535]), &destination, false);
        assert!(black[0] < 0.0 && black[2] < 0.0);
        let white = evaluate(&profile(true, [65535, 65535, 0]), &destination, false);
        assert!(white[0] > 1.0 && white[2] > 1.0);
    }

    #[test]
    fn both_policies_reject_extended_device_input_without_clamping() {
        let source = profile(false, [0; 3]);
        for icc in [false, true] {
            let lut = CmykLut::new(&source, &ColorProfile::new_srgb(), icc)
                .unwrap()
                .unwrap();
            for value in [-f32::EPSILON, 1.0 + f32::EPSILON, f32::NAN, f32::INFINITY] {
                for channel in 0..4 {
                    let mut input = [0.5; 4];
                    input[channel] = value;
                    let mut output = [123.0; 3];
                    assert!(lut.transform(&input, &mut output).is_err());
                    assert_eq!(output, [123.0; 3]);
                }
            }
        }
    }

    #[test]
    fn djxl_table_samples_use_reciprocal_and_exact_endpoint_lookup() {
        let table = LutStore::Store16(vec![65535, 257, 1]);
        assert_eq!(value(&table, 1, false), 257.0 * (1.0 / 65535.0));
        assert_ne!(value(&table, 1, false), value(&table, 1, true));
        assert_eq!(curve(&table, 3, 0, 1.0, false), 1.0 / 65535.0);
        assert_eq!(table_indices(2.0, 3, false), (2, 2));
        assert_eq!(table_indices(1.0, 3, false), (1, 1));
        assert_eq!(table_indices(0.5, 3, false), (0, 1));
    }

    #[test]
    fn djxl_clut_matches_source_corner_and_weight_order() {
        let mut source = profile(false, [0; 3]);
        let Some(LutWarehouse::Lut(lut)) = &mut source.lut_a_to_b_perceptual else {
            unreachable!()
        };
        lut.clut_table = LutStore::Store16(
            (0u32..48)
                .map(|i| (((i * 13817 + 991) ^ ((i + 5) * 1777)) & 65535) as u16)
                .collect(),
        );
        let lut = CmykLut::new(&source, &ColorProfile::new_srgb(), false)
            .unwrap()
            .unwrap();
        let mut output = [0.0; 3];
        lut.transform(&[0.1234567, 0.50001, 0.9001, 0.33333], &mut output)
            .unwrap();
        // Scalar Transform_inl.h::clut reference, before PCS or gamut math.
        assert_eq!(
            output.map(f32::to_bits),
            [0x3ecf_0af9, 0x3ecb_705b, 0x3f06_dc54]
        );
    }

    #[test]
    fn absolute_uses_media_white_not_header_white() {
        let mut source = profile(false, [0; 3]);
        let mut destination = ColorProfile::new_srgb();
        destination.rendering_intent = RenderingIntent::AbsoluteColorimetric;
        source.media_white_point = Some(Xyzd::new(0.4821, 0.5, 0.41245));
        destination.media_white_point = Some(Xyzd::new(0.9642, 1.0, 0.8249));
        assert_eq!(absolute_scale(&source, &destination).unwrap(), [0.5; 3]);
        source.media_white_point = None;
        source.white_point = Xyzd::new(0.3, 0.3, 0.3);
        assert_eq!(absolute_scale(&source, &destination).unwrap(), [1.0; 3]);
    }
}
