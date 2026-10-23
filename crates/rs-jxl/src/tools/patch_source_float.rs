// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! Floating-color Add path from frozen b866b3d enc_patch_dictionary.cc:
//! FindTextLikePatches stores f32 differences, RoundtripPatchFrame converts the
//! atlas through its declared format, SubtractFrom uses that reconstructed
//! atlas, and blending.cc adds it in dictionary order. base/float.h::LoadFloatRow
//! does NOT apply an integer normalization factor to floating-point samples.
//!
//! The source <=16-bit gate admits small floats, but not every Add candidate is
//! representable or pixel-lossless. enc_modular.cc::float_to_int can fail on a
//! delta's mantissa/exponent, and finite subtraction/addition can lose a signed
//! zero or a small addend even when both encoded planes are representable.
//! Atlas, residual and replay conversion all use the strict source inverse
//! bit check, including source subnormal/NaN mapping corner cases. Conversion
//! and replay failures are explicit errors here, never a claimed
//! lossless plan. A caller may retry without patches, but that is not identical
//! source candidate acceptance. Uncovered raw words are retained verbatim,
//! including NaN payloads/subnormals which a needless float re-encode may alter.

use super::{
    DetectionSamples, DetectionView, PatchBlendMode, PatchDictionary, REFERENCE_ID,
    SourcePatchConfig, Work, detect_view, invalid, pack_dictionary, source_enabled,
    source_supported, zeros,
};
use crate::modular::sample_convert::{SampleFormat, float_to_int_lossless, int_to_float};
use crate::{Error, Result};

/// Color only, one gray or three RGB planes, with words in the DECLARED float
/// format. These are not u16 values normalized to [0,1], nor f32 bit patterns
/// unless the declared format is binary32. Extras stay in the frame adapter.
#[derive(Clone, Copy, Debug)]
pub struct FloatImageView<'a> {
    pub width: usize,
    pub height: usize,
    pub format: SampleFormat,
    pub planes: &'a [Vec<i32>],
}

impl FloatImageView<'_> {
    fn validate(self) -> Result<usize> {
        self.format.validate()?;
        if !matches!(self.format, SampleFormat::Float { .. }) {
            return Err(invalid(
                "float patches require a declared floating-point format",
            ));
        }
        if self.width == 0
            || self.height == 0
            || self.width > i32::MAX as usize
            || self.height > i32::MAX as usize
            || !matches!(self.planes.len(), 1 | 3)
        {
            return Err(invalid(
                "invalid float patch color dimensions or plane count",
            ));
        }
        let count = self
            .width
            .checked_mul(self.height)
            .ok_or(Error::Limit("float patch pixels"))?;
        if self.planes.iter().any(|plane| plane.len() != count) {
            return Err(invalid("invalid float patch plane size"));
        }
        Ok(count)
    }

    fn decode(self) -> Result<Vec<Vec<f32>>> {
        let count = self.validate()?;
        self.planes
            .iter()
            .map(|plane| {
                let mut values = zeros(count)?;
                // Source inverse constructs f32 bits, preserving signs, payloads,
                // and subnormals without interpreting word magnitudes as samples.
                int_to_float(plane, &mut values, self.format, 1.0)?;
                Ok(values)
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FloatPatchImage {
    pub width: usize,
    pub height: usize,
    pub format: SampleFormat,
    /// Encoded Modular float words, not signed arithmetic residual integers.
    pub planes: Vec<Vec<i32>>,
}

impl FloatPatchImage {
    pub fn view(&self) -> FloatImageView<'_> {
        FloatImageView {
            width: self.width,
            height: self.height,
            format: self.format,
            planes: &self.planes,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FloatPatchPlan {
    pub reference: FloatPatchImage,
    pub residual: FloatPatchImage,
    pub dictionary: PatchDictionary,
}

struct Applied {
    values: Vec<Vec<f32>>,
    covered: Vec<u8>,
}

fn apply(
    image: FloatImageView<'_>,
    reference: FloatImageView<'_>,
    dictionary: &PatchDictionary,
    subtract: bool,
) -> Result<Applied> {
    let count = image.validate()?;
    reference.validate()?;
    let stride = dictionary.validate()?;
    if image.format != reference.format || image.planes.len() != reference.planes.len() {
        return Err(invalid("float patch reference metadata differs"));
    }
    for blends in dictionary.blendings.chunks_exact(stride) {
        if !matches!(blends[0].mode, PatchBlendMode::None | PatchBlendMode::Add)
            || blends[1..].iter().any(|b| b.mode != PatchBlendMode::None)
        {
            return Err(invalid(
                "float Add patches modify color only; extras must use None",
            ));
        }
    }
    let runs = dictionary
        .positions
        .iter()
        .enumerate()
        .filter(|&(i, position)| {
            i == 0 || position.ref_pos_idx != dictionary.positions[i - 1].ref_pos_idx
        })
        .count();
    let max_refs = 1024usize
        .checked_add(count / 4)
        .ok_or(Error::Limit("float patch count"))?;
    if runs > max_refs || dictionary.positions.len() > max_refs.saturating_mul(4) {
        return Err(Error::Limit("float patches exceed decoder count limits"));
    }
    let mut values = image.decode()?;
    let reference_values = reference.decode()?;
    let mut covered = zeros(count)?;
    // dec_patch_dictionary.cc::GetPatchesForRow sorts the returned indices;
    // position-major traversal therefore preserves each pixel's source order.
    for (index, position) in dictionary.positions.iter().enumerate() {
        let crop = dictionary.references[position.ref_pos_idx];
        if crop.reference != REFERENCE_ID
            || u64::from(crop.x0) + u64::from(crop.width) > reference.width as u64
            || u64::from(crop.y0) + u64::from(crop.height) > reference.height as u64
            || u64::from(position.x) + u64::from(crop.width) > image.width as u64
            || u64::from(position.y) + u64::from(crop.height) > image.height as u64
        {
            return Err(invalid("float patch rectangle outside reference or image"));
        }
        if dictionary.blendings[index * stride].mode == PatchBlendMode::None {
            continue;
        }
        for y in 0..crop.height as usize {
            for x in 0..crop.width as usize {
                let dst = (position.y as usize + y) * image.width + position.x as usize + x;
                let src = (crop.y0 as usize + y) * reference.width + crop.x0 as usize + x;
                covered[dst] = 1;
                for (plane, atlas) in values.iter_mut().zip(&reference_values) {
                    plane[dst] = if subtract {
                        plane[dst] - atlas[src]
                    } else {
                        plane[dst] + atlas[src]
                    };
                }
            }
        }
    }
    Ok(Applied { values, covered })
}

fn conversion_error(stage: &str, error: Error) -> Error {
    Error::Encoding(format!(
        "source float patch {stage} is not representable: {error}"
    ))
}

fn encode_covered(
    image: FloatImageView<'_>,
    applied: &Applied,
    stage: &str,
) -> Result<Vec<Vec<i32>>> {
    let mut words = image.planes.to_vec();
    for (output, values) in words.iter_mut().zip(&applied.values) {
        for (position, &covered) in applied.covered.iter().enumerate() {
            if covered == 0 {
                continue;
            }
            let value = &values[position..position + 1];
            let out = &mut output[position..position + 1];
            float_to_int_lossless(value, out, image.format)
                .map_err(|e| conversion_error(stage, e))?;
        }
    }
    Ok(words)
}

/// Decode the stored float words, add in source order, and convert covered
/// pixels back without rounding. Uncovered sample words are not re-encoded.
pub fn compose_float_add(
    residual: &FloatPatchImage,
    reference: &FloatPatchImage,
    dictionary: &PatchDictionary,
) -> Result<Vec<Vec<i32>>> {
    let applied = apply(residual.view(), reference.view(), dictionary, false)?;
    encode_covered(residual.view(), &applied, "replay")
}

/// Executable source float detector/atlas/residual path, with a mandatory
/// pixel-bit replay guard. None is a source gate, no candidates, or atlas budget;
/// format-conversion and nonexact-replay failures are explicit errors.
pub fn discover_source_float(
    image: FloatImageView<'_>,
    config: &SourcePatchConfig,
) -> Result<Option<FloatPatchPlan>> {
    config.validate()?;
    image.format.validate()?;
    if !matches!(image.format, SampleFormat::Float { .. }) {
        return Err(invalid(
            "float patches require a declared floating-point format",
        ));
    }
    if !config.enabled
        || !source_enabled(
            u32::from(image.format.bits()),
            config.speed_tier,
            config.decoding_speed_tier,
            config.do_color,
            config.groupwise,
        )
        || !source_supported(
            u32::from(image.format.bits()),
            true,
            &config.extra_channel_dim_shifts,
        )
    {
        return Ok(None);
    }
    let values = image.decode()?;
    let info = detect_view(
        DetectionView {
            width: image.width,
            height: image.height,
            color_channels: values.len(),
            samples: DetectionSamples::Float(&values),
        },
        config,
        &mut Work::default(),
    )?;
    let Some(packed) = pack_dictionary(&info, values.len(), 0, config.limits.max_reference_pixels)?
    else {
        return Ok(None);
    };
    let mut reference = FloatPatchImage {
        width: packed.width,
        height: packed.height,
        format: image.format,
        planes: packed.planes,
    };
    for plane in &mut reference.planes {
        // Detection stores f32 delta bits, NOT declared-format float words.
        // RoundtripPatchFrame encodes those values then decodes the reference
        // before SubtractFrom; apply() performs that exact source inverse.
        let deltas: Vec<_> = plane
            .iter()
            .map(|&bits| f32::from_bits(bits as u32))
            .collect();
        float_to_int_lossless(&deltas, plane, image.format)
            .map_err(|e| conversion_error("reference", e))?;
    }
    let applied = apply(image, reference.view(), &packed.dictionary, true)?;
    let residual = FloatPatchImage {
        width: image.width,
        height: image.height,
        format: image.format,
        planes: encode_covered(image, &applied, "residual")?,
    };
    let replayed = compose_float_add(&residual, &reference, &packed.dictionary)?;
    for (channel, (actual, expected)) in replayed.iter().zip(image.planes).enumerate() {
        if let Some(position) = actual.iter().zip(expected).position(|(a, b)| a != b) {
            return Err(Error::Encoding(format!(
                "source float patch Add is not bit-exact at channel {channel}, sample {position}"
            )));
        }
    }
    Ok(Some(FloatPatchPlan {
        reference,
        residual,
        dictionary: packed.dictionary,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::{PatchBlending, PatchPosition, PatchReferencePosition};
    use super::*;
    use crate::modular::sample_convert::float_to_int;

    const WIDTH: usize = 129;
    const HEIGHT: usize = 83;
    const ORIGINS: [(usize, usize); 4] = [(17, 17), (81, 17), (33, 53), (97, 53)];

    fn fixture(
        format: SampleFormat,
        colors: usize,
        background: [f32; 2],
        ink: [f32; 2],
    ) -> FloatPatchImage {
        let mut values = Vec::new();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let side = usize::from(x >= WIDTH / 2);
                let glyph = ORIGINS
                    .iter()
                    .any(|&(x0, y0)| (x0..x0 + 6).contains(&x) && (y0..y0 + 5).contains(&y));
                values.push(if glyph { ink[side] } else { background[side] });
            }
        }
        let mut words = vec![0; values.len()];
        float_to_int_lossless(&values, &mut words, format).unwrap();
        FloatPatchImage {
            width: WIDTH,
            height: HEIGHT,
            format,
            planes: vec![words; colors],
        }
    }

    fn assert_plan(image: &FloatPatchImage, config: &SourcePatchConfig) -> FloatPatchPlan {
        let plan = discover_source_float(image.view(), config)
            .unwrap()
            .expect("active float fixture");
        assert_eq!(plan.dictionary.positions.len(), 4);
        assert_eq!(plan.dictionary.references.len(), 1);
        assert_eq!(plan.dictionary.num_extra_channels, 0);
        assert!(
            plan.dictionary
                .blendings
                .iter()
                .all(|b| b.mode == PatchBlendMode::Add)
        );
        assert_eq!(
            compose_float_add(&plan.residual, &plan.reference, &plan.dictionary).unwrap(),
            image.planes
        );
        assert!(
            plan.reference
                .view()
                .decode()
                .unwrap()
                .iter()
                .flatten()
                .any(|&v| v == -0.5)
        );
        super::super::super::tokenize(&plan.dictionary).unwrap();
        plan
    }

    #[test]
    fn all_small_float_formats_use_real_values_and_exact_add_replay() {
        for exponent_bits in 2..=8 {
            for bits in exponent_bits + 3..=16 {
                let format = SampleFormat::Float {
                    bits,
                    exponent_bits,
                };
                for colors in [1, 3] {
                    let image = fixture(format, colors, [1.0, 1.5], [0.5, 1.0]);
                    let config = SourcePatchConfig {
                        speed_tier: -1,
                        floating_point: true,
                        ..Default::default()
                    };
                    let plan = assert_plan(&image, &config);
                    // Both occurrences share -0.5 despite different absolute
                    // backgrounds. No unsigned/max normalization was applied.
                    let atlas = plan.reference.view().decode().unwrap();
                    assert!(atlas.iter().flatten().all(|&v| v == 0.0 || v == -0.5));
                }
            }
        }
    }

    #[test]
    fn fast_and_baseline_float_detectors_match_and_uncovered_special_words_survive() {
        let mut image = fixture(SampleFormat::BINARY16, 3, [1.0, 1.5], [0.5, 1.0]);
        // Outside every accepted rectangle: negative zero, sNaN/qNaN payloads,
        // infinities and the smallest subnormal must not be re-encoded.
        let special = [0x8000, 0x7d31, 0xfe23, 0x7c00, 0xfc00, 1];
        for plane in &mut image.planes {
            plane[..special.len()].copy_from_slice(&special);
        }
        let original = image.clone();
        let fast = assert_plan(&image, &SourcePatchConfig::default());
        let slow = assert_plan(
            &image,
            &SourcePatchConfig {
                fast: false,
                ..Default::default()
            },
        );
        assert_eq!(fast.dictionary, slow.dictionary);
        assert_eq!(fast.reference, slow.reference);
        assert_eq!(fast.residual, slow.residual);
        for plane in &fast.residual.planes {
            assert_eq!(&plane[..special.len()], &special);
        }
        assert_eq!(image, original);
    }

    #[test]
    fn source_reference_and_residual_precision_failures_are_not_silently_rounded() {
        let quarter_ulp = 1.0f32 / 4096.0;
        let atlas = fixture(SampleFormat::BINARY16, 1, [1.0; 2], [quarter_ulp; 2]);
        // Source f32 delta -0.999755859375 lies between adjacent F16 values.
        assert!(float_to_int(&[quarter_ulp - 1.0], &mut [0], SampleFormat::BINARY16, 1.0).is_err());
        let error = discover_source_float(atlas.view(), &SourcePatchConfig::default()).unwrap_err();
        assert!(error.to_string().contains("reference is not representable"));

        let residual = fixture(SampleFormat::BINARY16, 1, [1.0; 2], [0.0, 1.0 / 2048.0]);
        // Representative atlas=-1 is exact. The other occurrence's residual
        // 1.00048828125 is not F16, as source float_to_int's mantissa check says.
        assert!(
            float_to_int(&[1.0 + 1.0 / 2048.0], &mut [0], SampleFormat::BINARY16, 1.0).is_err()
        );
        let error =
            discover_source_float(residual.view(), &SourcePatchConfig::default()).unwrap_err();
        assert!(error.to_string().contains("residual is not representable"));
    }

    #[test]
    fn source_subtract_add_loses_negative_zero_and_small_addend_even_with_exact_storage() {
        for lost_word in [0x8000, 1] {
            let mut image = fixture(SampleFormat::BINARY16, 1, [1.0; 2], [0.0; 2]);
            let at = ORIGINS[1].1 * WIDTH + ORIGINS[1].0;
            image.planes[0][at] = lost_word;
            let value = image.view().decode().unwrap()[0][at];
            let reference = -1.0f32;
            let residual = value - reference;
            let mut words = [0];
            float_to_int(&[residual], &mut words, image.format, 1.0).unwrap();
            let mut decoded = [0.0];
            int_to_float(&words, &mut decoded, image.format, 1.0).unwrap();
            // enc_patch_dictionary.cc:171 and blending.cc:119, no approximation
            // or lossy codec needed: these exact source operations lose bits.
            assert_eq!((decoded[0] + reference).to_bits(), 0);
            assert_ne!((decoded[0] + reference).to_bits(), value.to_bits());
            let error =
                discover_source_float(image.view(), &SourcePatchConfig::default()).unwrap_err();
            assert!(error.to_string().contains("not bit-exact"));
        }
    }

    fn one_pixel_dictionary(count: usize) -> PatchDictionary {
        PatchDictionary {
            references: (0..count)
                .map(|x| PatchReferencePosition {
                    reference: REFERENCE_ID,
                    x0: x as u32,
                    y0: 0,
                    width: 1,
                    height: 1,
                })
                .collect(),
            positions: (0..count)
                .map(|ref_pos_idx| PatchPosition {
                    x: 0,
                    y: 0,
                    ref_pos_idx,
                })
                .collect(),
            blendings: vec![
                PatchBlending {
                    mode: PatchBlendMode::Add,
                    alpha_channel: 0,
                    clamp: false
                };
                count
            ],
            num_extra_channels: 0,
        }
    }

    #[test]
    fn overlapping_float_patches_keep_dictionary_order_and_validate_crops() {
        let format = SampleFormat::Float {
            bits: 16,
            exponent_bits: 8,
        };
        let residual = FloatPatchImage {
            width: 1,
            height: 1,
            format,
            planes: vec![vec![0]],
        };
        let mut words = vec![0; 3];
        float_to_int(&[16_777_216.0, 1.0, -16_777_216.0], &mut words, format, 1.0).unwrap();
        let reference = FloatPatchImage {
            width: 3,
            height: 1,
            format,
            planes: vec![words],
        };
        let mut dictionary = one_pixel_dictionary(3);
        assert_eq!(
            compose_float_add(&residual, &reference, &dictionary).unwrap(),
            [vec![0]]
        );
        dictionary.positions.swap(1, 2);
        let alternate = compose_float_add(&residual, &reference, &dictionary).unwrap();
        let mut decoded = [0.0];
        int_to_float(&alternate[0], &mut decoded, format, 1.0).unwrap();
        assert_eq!(decoded, [1.0]);
        dictionary.positions[0].x = 1;
        assert!(compose_float_add(&residual, &reference, &dictionary).is_err());
        dictionary.positions[0].x = 0;
        dictionary.blendings[0].mode = PatchBlendMode::Replace;
        assert!(compose_float_add(&residual, &reference, &dictionary).is_err());
    }

    #[test]
    fn typed_format_input_and_source_gates_remain_explicit() {
        let input = fixture(SampleFormat::BINARY16, 1, [1.0, 1.5], [0.5, 1.0]);
        for config in [
            SourcePatchConfig {
                speed_tier: 6,
                ..Default::default()
            },
            SourcePatchConfig {
                decoding_speed_tier: 2,
                ..Default::default()
            },
            SourcePatchConfig {
                groupwise: true,
                ..Default::default()
            },
        ] {
            assert!(
                discover_source_float(input.view(), &config)
                    .unwrap()
                    .is_none()
            );
        }
        let config = SourcePatchConfig::default();
        let mut view = input.view();
        view.format = SampleFormat::BINARY32;
        assert!(discover_source_float(view, &config).unwrap().is_none());
        view.format = SampleFormat::Unsigned { bits: 16 };
        assert!(discover_source_float(view, &config).is_err());
        view = input.view();
        view.width += 1;
        assert!(discover_source_float(view, &config).is_err());
        let mut bad = input.clone();
        bad.planes[0][0] = 65536;
        assert!(discover_source_float(bad.view(), &config).is_err());
    }
}
