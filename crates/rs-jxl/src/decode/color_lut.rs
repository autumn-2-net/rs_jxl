//! Direct evaluation of parsed ICC mft1/mft2 CMYK tables. Keeping input curves,
//! CLUT interpolation, output curves and PCS conversion in that order avoids
//! resampling a nonlinear Lab-to-XYZ transform into a second coarse grid.
//! ICC byte parsing remains the responsibility of moxcms.

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
}

impl CmykLut {
    pub(super) fn new(source: &ColorProfile, destination: &ColorProfile) -> Result<Option<Self>> {
        if source.color_space != DataColorSpace::Cmyk {
            return Ok(None);
        }
        let selected = match destination.rendering_intent {
            RenderingIntent::Perceptual => &source.lut_a_to_b_perceptual,
            RenderingIntent::Saturation => &source.lut_a_to_b_saturation,
            _ => &source.lut_a_to_b_colorimetric,
        };
        let Some(LutWarehouse::Lut(lut)) =
            selected.as_ref().or(source.lut_a_to_b_perceptual.as_ref())
        else {
            return Ok(None);
        };
        if lut.num_input_channels != 4
            || lut.num_output_channels != 3
            || !matches!(lut.lut_type, LutType::Lut8 | LutType::Lut16)
            || !matches!(source.pcs, DataColorSpace::Lab | DataColorSpace::Xyz)
        {
            return Ok(None);
        }
        let n = usize::from(lut.num_clut_grid_points);
        let input_entries = usize::from(lut.num_input_table_entries);
        let output_entries = usize::from(lut.num_output_table_entries);
        if n < 2
            || input_entries < 2
            || output_entries < 2
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
            lab_scale: if lut.lut_type == LutType::Lut16 {
                65535.0 / 65280.0
            } else {
                1.0
            },
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
            let mut fraction = [0.0; 4];
            for c in 0..4 {
                let v = curve(
                    &self.lut.input_table,
                    usize::from(self.lut.num_input_table_entries),
                    c,
                    input[c],
                );
                let position = v * (grid - 1) as f32;
                low[c] = (position as usize).min(grid - 2);
                fraction[c] = position - low[c] as f32;
            }
            let mut pcs = [0.0; 3];
            for corner in 0..16 {
                let mut index = 0;
                let mut weight = 1.0;
                for c in 0..4 {
                    let high = (corner >> (3 - c)) & 1;
                    index = index * grid + low[c] + high;
                    weight *= if high == 0 {
                        1.0 - fraction[c]
                    } else {
                        fraction[c]
                    };
                }
                for (c, v) in pcs.iter_mut().enumerate() {
                    *v += weight * value(&self.lut.clut_table, index * 3 + c);
                }
            }
            for (c, v) in pcs.iter_mut().enumerate() {
                *v = curve(
                    &self.lut.output_table,
                    usize::from(self.lut.num_output_table_entries),
                    c,
                    *v,
                );
            }
            let xyz = if self.lab {
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
            } else {
                pcs.map(|v| v * (65535.0 / 32768.0))
            };
            output.copy_from_slice(&xyz);
        }
        Ok(())
    }
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
fn value(table: &LutStore, index: usize) -> f32 {
    match table {
        LutStore::Store8(v) => f32::from(v[index]) / 255.0,
        LutStore::Store16(v) => f32::from(v[index]) / 65535.0,
    }
}
fn curve(table: &LutStore, entries: usize, channel: usize, input: f32) -> f32 {
    let position = input.clamp(0.0, 1.0) * (entries - 1) as f32;
    let low = (position as usize).min(entries - 2);
    let a = value(table, channel * entries + low);
    let b = value(table, channel * entries + low + 1);
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
            let lut = CmykLut::new(&profile(true, [l, 32768, 32768]), &destination)
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
        let lut = CmykLut::new(&p, &ColorProfile::new_srgb())
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
        let lut = CmykLut::new(&p, &ColorProfile::new_srgb())
            .unwrap()
            .unwrap();
        assert!(lut.transform(&[0.0; 3], &mut [0.0; 3]).is_err());
        assert!(lut.transform(&[f32::NAN; 4], &mut [0.0; 3]).is_err());
        assert!(lut.transform(&[-0.1; 4], &mut [0.0; 3]).is_err());
        let Some(LutWarehouse::Lut(table)) = &mut p.lut_a_to_b_perceptual else {
            unreachable!()
        };
        table.num_clut_grid_points = 255;
        assert!(CmykLut::new(&p, &ColorProfile::new_srgb()).is_err());
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
