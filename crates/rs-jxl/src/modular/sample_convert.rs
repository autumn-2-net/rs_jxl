// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

//! Sample mappings from libjxl b866b3defec13494bfbaf705db93581cda118e37:
//! - `lib/jxl/enc_modular.cc::float_to_int` (lines 193..275).
//! - `lib/jxl/dec_modular.cc::{int_to_float, SingleFromSingleAccurate}` and
//!   `SingleFromSingle` for the inverse reference.
//! - `lib/jxl/image_metadata.cc::BitDepth::VisitFields` for format limits.
//!
//! `samples_to_int` is the exact typed-input route: integers stay integers,
//! while float samples are supplied as raw words, including signed zeros and
//! all NaN payloads. F16 words use u16; binary32/custom words use u32. No float
//! arithmetic or sign extension is involved. A binary32 word with bit 31 set
//! becomes a negative i32; smaller float formats remain nonnegative bit words.
//!
//! `float_to_int` intentionally retains SOURCE behavior. For custom floats,
//! NaN payload low bits are discarded without a precision check and a tiny
//! signaling NaN can become infinity. The source shifts subnormal mantissas
//! before checking precision; it can discard low bits there too. For exponent
//! width 8 and total width < 32, it also inserts an implicit leading one into
//! binary32 subnormals. Do not assume every successful source conversion is
//! bit-exact: `float_to_int_lossless` checks the source inverse bit-for-bit and
//! rejects these cases. Raw-word mapping avoids them altogether.
//!
//! Normalized integer conversion is not the u31 lossless input route: f32
//! cannot represent all u31 values. It preserves the source's f32 arithmetic
//! through 22 bits and f64 arithmetic above that, with signed half-away-from-
//! zero adjustment and truncation. There is no clipping to the declared range.
//! Undefined C++ casts (nonfinite/outside i32 after truncation) are errors,
//! never Rust's saturating float-to-int casts. No FMA is used explicitly.
//!
//! Row slices must have equal lengths. Format/length/scale errors do not write
//! output. Sample errors leave the successfully converted prefix written and
//! the failing element/suffix untouched, matching the source row convention.
//! This module does not register itself or change Image/metadata. Dedicated
//! SIMD, channel preparation, and codestream precision policy are not included.
//! Core boundary: PackSigned accepts a full i32, not an arbitrary i64 residual.
//! Float32 bit words and u31 values therefore fit Zero/identity tokens, but
//! predictors, offsets, and transforms can exceed the checked residual domain.

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleFormat {
    Unsigned { bits: u8 },
    Float { bits: u8, exponent_bits: u8 },
}

impl SampleFormat {
    pub const BINARY16: Self = Self::Float {
        bits: 16,
        exponent_bits: 5,
    };
    pub const BINARY32: Self = Self::Float {
        bits: 32,
        exponent_bits: 8,
    };

    pub const fn bits(self) -> u8 {
        match self {
            Self::Unsigned { bits } | Self::Float { bits, .. } => bits,
        }
    }

    /// Metadata limits, before shifts or float-layout arithmetic are performed.
    pub fn validate(self) -> Result<()> {
        let valid = match self {
            Self::Unsigned { bits } => (1..=31).contains(&bits),
            Self::Float {
                bits,
                exponent_bits,
            } => {
                let mantissa = i32::from(bits) - i32::from(exponent_bits) - 1;
                (2..=8).contains(&exponent_bits) && (2..=23).contains(&mantissa)
            }
        };
        if !valid {
            return Err(Error::InvalidInput("invalid modular sample format".into()));
        }
        Ok(())
    }

    fn float_layout(self) -> Result<FloatLayout> {
        self.validate()?;
        match self {
            Self::Float {
                bits,
                exponent_bits,
            } => Ok(FloatLayout::new(bits, exponent_bits)),
            Self::Unsigned { .. } => Err(Error::InvalidInput(
                "bit-exact float conversion requires a float sample format".into(),
            )),
        }
    }

    // Call only after validate(); float metadata implies at most 32 total bits.
    fn word_mask(self) -> u32 {
        u32::MAX >> (32 - u32::from(self.bits()))
    }
}

fn check_lengths(input: usize, output: usize) -> Result<()> {
    if input != output {
        return Err(Error::InvalidInput("sample row length mismatch".into()));
    }
    Ok(())
}

fn sample_error(index: usize, reason: &str) -> Error {
    Error::InvalidInput(format!("sample {index}: {reason}"))
}

fn check_factor(bits: u8, factor: f64) -> Result<()> {
    if !factor.is_finite() || (bits <= 22 && !(factor as f32).is_finite()) {
        return Err(Error::InvalidInput(
            "nonfinite sample conversion factor".into(),
        ));
    }
    Ok(())
}

/// Direct source FloatToInt. Float formats ignore factor, including NaN/Inf.
/// Unsigned formats normally use factor = 2^bits - 1, but accept other finite
/// factors for source-style scaling. Use samples_to_int for exact u31 inputs.
pub fn float_to_int(
    input: &[f32],
    output: &mut [i32],
    format: SampleFormat,
    factor: f64,
) -> Result<()> {
    check_lengths(input.len(), output.len())?;
    format.validate()?;
    match format {
        SampleFormat::Float { .. } => {
            let layout = format.float_layout()?;
            for (index, (&value, out)) in input.iter().zip(output).enumerate() {
                *out = layout.encode(value.to_bits(), index)?;
            }
        }
        SampleFormat::Unsigned { bits } => {
            check_factor(bits, factor)?;
            for (index, (&value, out)) in input.iter().zip(output).enumerate() {
                // Keep the precision split and original-input sign test: do
                // not replace this with round(), mul_add(), or signum().
                let adjusted = if bits > 22 {
                    f64::from(value) * factor + if value < 0.0 { -0.5 } else { 0.5 }
                } else {
                    f64::from(value * factor as f32 + if value < 0.0 { -0.5f32 } else { 0.5f32 })
                };
                let integer = adjusted.trunc();
                if !integer.is_finite()
                    || integer < f64::from(i32::MIN)
                    || integer > f64::from(i32::MAX)
                {
                    return Err(sample_error(index, "nonfinite or out-of-i32 scaled sample"));
                }
                *out = integer as i32;
            }
        }
    }
    Ok(())
}

/// Source float conversion plus a strict source-inverse bit-pattern check.
/// Accepts only Float formats. Signed zeros, NaN sign/payload/signaling bits,
/// and subnormal values must all survive exactly; there is no NaN exception.
pub fn float_to_int_lossless(
    input: &[f32],
    output: &mut [i32],
    format: SampleFormat,
) -> Result<()> {
    check_lengths(input.len(), output.len())?;
    let layout = format.float_layout()?;
    for (index, (&value, out)) in input.iter().zip(output).enumerate() {
        let bits = value.to_bits();
        let encoded = layout.encode(bits, index)?;
        if layout.decode(encoded, index)? != bits {
            return Err(sample_error(
                index,
                "source float mapping changes the input bit pattern",
            ));
        }
        *out = encoded;
    }
    Ok(())
}

/// Source inverse reference. For Float formats the factor is ignored and the
/// output is assembled from raw bits, including signaling NaNs. For Unsigned
/// formats, supply the inverse factor (normally 1 / (2^bits - 1)); this f32
/// reference is not an exact integer round-trip for arbitrary u31 samples.
pub fn int_to_float(
    input: &[i32],
    output: &mut [f32],
    format: SampleFormat,
    inverse_factor: f64,
) -> Result<()> {
    check_lengths(input.len(), output.len())?;
    format.validate()?;
    match format {
        SampleFormat::Float { .. } => {
            let layout = format.float_layout()?;
            for (index, (&value, out)) in input.iter().zip(output).enumerate() {
                *out = f32::from_bits(layout.decode(value, index)?);
            }
        }
        SampleFormat::Unsigned { bits } => {
            check_factor(bits, inverse_factor)?;
            for (&value, out) in input.iter().zip(output) {
                *out = if bits > 22 {
                    (f64::from(value) * inverse_factor) as f32
                } else {
                    value as f32 * inverse_factor as f32
                };
            }
        }
    }
    Ok(())
}

/// Exact mapping of typed unsigned samples or raw floating-point words.
/// Accepts u8/u16/u32 slices. Bits outside the declared format are errors.
/// This is the recommended pixel-lossless input path, including custom floats.
pub fn samples_to_int<T: Copy + Into<u32>>(
    input: &[T],
    output: &mut [i32],
    format: SampleFormat,
) -> Result<()> {
    check_lengths(input.len(), output.len())?;
    format.validate()?;
    let mask = format.word_mask();
    for (index, (&value, out)) in input.iter().zip(output).enumerate() {
        let word = value.into();
        if word & !mask != 0 {
            return Err(sample_error(
                index,
                "sample word exceeds declared bit depth",
            ));
        }
        *out = word as i32;
    }
    Ok(())
}

/// Exact inverse of samples_to_int. Floating-point outputs are raw words, not
/// numerical conversions; compare these words for pixel-lossless acceptance.
pub fn int_to_samples(input: &[i32], output: &mut [u32], format: SampleFormat) -> Result<()> {
    check_lengths(input.len(), output.len())?;
    format.validate()?;
    let mask = format.word_mask();
    for (index, (&value, out)) in input.iter().zip(output).enumerate() {
        let word = value as u32;
        if word & !mask != 0 {
            return Err(sample_error(
                index,
                "modular sample exceeds declared bit depth",
            ));
        }
        *out = word;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct FloatLayout {
    bits: u8,
    exponent_bits: u8,
    mantissa_bits: u32,
    mantissa_shift: u32,
    exponent_bias: i32,
    exponent_max: i32,
    sign: u32,
    word_mask: u32,
}

impl FloatLayout {
    // Metadata validation precedes construction; every shift is in 0..=31.
    fn new(bits: u8, exponent_bits: u8) -> Self {
        let mantissa_bits = u32::from(bits - exponent_bits - 1);
        Self {
            bits,
            exponent_bits,
            mantissa_bits,
            mantissa_shift: 23 - mantissa_bits,
            exponent_bias: (1 << (exponent_bits - 1)) - 1,
            exponent_max: (1 << exponent_bits) - 1,
            sign: 1u32 << (bits - 1),
            word_mask: u32::MAX >> (32 - u32::from(bits)),
        }
    }

    fn encode(self, bits: u32, index: usize) -> Result<i32> {
        if self.bits == 32 {
            return Ok(bits as i32);
        }
        let sign = if bits >> 31 != 0 { self.sign } else { 0 };
        let magnitude = bits & 0x7fff_ffff;
        if magnitude == 0 {
            return Ok(sign as i32);
        }
        let mut exponent = (magnitude >> 23) as i32 - 127;
        let mut mantissa = magnitude & 0x007f_ffff;
        if exponent == 128 {
            return Ok((sign
                | (self.exponent_max as u32) << self.mantissa_bits
                | mantissa >> self.mantissa_shift) as i32);
        }
        exponent += self.exponent_bias;
        if exponent <= 0 {
            // Preserve the source's implicit-one and shift-before-check order,
            // even for a binary32 subnormal. The strict entry point detects
            // accepted inputs that this source operation changes.
            mantissa |= 0x0080_0000;
            if exponent < -(self.mantissa_bits as i32) {
                return Err(sample_error(
                    index,
                    "float exponent underflows the custom format",
                ));
            }
            mantissa >>= (1 - exponent) as u32;
            exponent = 0;
        }
        if exponent >= self.exponent_max {
            return Err(sample_error(
                index,
                "float exponent overflows the custom format",
            ));
        }
        if mantissa & ((1u32 << self.mantissa_shift) - 1) != 0 {
            return Err(sample_error(index, "float mantissa loses precision"));
        }
        Ok(
            (sign | (exponent as u32) << self.mantissa_bits | mantissa >> self.mantissa_shift)
                as i32,
        )
    }

    fn decode(self, sample: i32, index: usize) -> Result<u32> {
        let word = sample as u32;
        if self.bits == 32 {
            return Ok(word);
        }
        if word & !self.word_mask != 0 {
            return Err(sample_error(index, "float word exceeds declared bit depth"));
        }
        let sign = if word & self.sign != 0 {
            0x8000_0000
        } else {
            0
        };
        let magnitude = word & (self.sign - 1);
        if magnitude == 0 {
            return Ok(sign);
        }
        let mut exponent = (magnitude >> self.mantissa_bits) as i32;
        let mut mantissa = magnitude & ((1u32 << self.mantissa_bits) - 1);
        if exponent == self.exponent_max {
            return Ok(sign | 0x7f80_0000 | mantissa << self.mantissa_shift);
        }
        mantissa <<= self.mantissa_shift;
        if exponent == 0 && self.exponent_bits < 8 {
            while mantissa & 0x0080_0000 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            exponent += 1;
            mantissa &= 0x007f_ffff;
        }
        exponent += 127 - self.exponent_bias;
        if exponent < 0 {
            return Err(sample_error(
                index,
                "custom float cannot be represented as binary32",
            ));
        }
        Ok(sign | (exponent as u32) << 23 | mantissa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(words: &[u32], format: SampleFormat) -> Vec<f32> {
        let mut modular = vec![0; words.len()];
        samples_to_int(words, &mut modular, format).unwrap();
        let mut floats = vec![0.0; words.len()];
        int_to_float(&modular, &mut floats, format, f64::NAN).unwrap();
        floats
    }

    fn source_encode(bits: &[u32], format: SampleFormat) -> Result<Vec<i32>> {
        let floats: Vec<_> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        let mut out = vec![0; bits.len()];
        float_to_int(&floats, &mut out, format, f64::NAN)?;
        Ok(out)
    }

    #[test]
    fn binary32_bit_patterns_are_copied_without_float_arithmetic() {
        let mut words = vec![
            0,
            0x8000_0000,
            1,
            0x007f_ffff,
            0x0080_0000,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0x7fc0_0001,
            0xffff_ffff,
        ];
        let mut random = 0xb866_b3deu32;
        for _ in 0..4096 {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            words.push(random);
        }
        let floats = decode(&words, SampleFormat::BINARY32);
        assert_eq!(
            floats.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            words
        );
        let mut out = vec![0; words.len()];
        float_to_int_lossless(&floats, &mut out, SampleFormat::BINARY32).unwrap();
        assert_eq!(out, words.iter().map(|&v| v as i32).collect::<Vec<_>>());
        assert_eq!(source_encode(&words, SampleFormat::BINARY32).unwrap(), out);
        let mut inverse = vec![0; words.len()];
        int_to_samples(&out, &mut inverse, SampleFormat::BINARY32).unwrap();
        assert_eq!(inverse, words);
    }

    #[test]
    fn every_binary16_word_round_trips_including_signaling_nans() {
        let words: Vec<u16> = (0..=u16::MAX).collect();
        let mut modular = vec![0; words.len()];
        samples_to_int(&words, &mut modular, SampleFormat::BINARY16).unwrap();
        let mut floats = vec![0.0; words.len()];
        int_to_float(&modular, &mut floats, SampleFormat::BINARY16, 1.0).unwrap();
        let mut encoded = vec![0; words.len()];
        float_to_int_lossless(&floats, &mut encoded, SampleFormat::BINARY16).unwrap();
        assert_eq!(encoded, modular);
        for (word, value) in words.iter().copied().zip(floats) {
            let sign = u32::from(word & 0x8000) << 16;
            let exponent = (word >> 10) & 31;
            let mantissa = word & 1023;
            let expected = match exponent {
                0 => (f32::from(mantissa) * 2.0f32.powi(-24)).to_bits() | sign,
                31 => sign | 0x7f80_0000 | u32::from(mantissa) << 13,
                _ => sign | (u32::from(exponent) + 112) << 23 | u32::from(mantissa) << 13,
            };
            assert_eq!(value.to_bits(), expected, "word {word:04x}");
        }
    }

    #[test]
    fn custom_metadata_layouts_and_source_inverse() {
        for exponent_bits in 2..=8u8 {
            for mantissa_bits in 2..=23u8 {
                let bits = 1 + exponent_bits + mantissa_bits;
                let format = SampleFormat::Float {
                    bits,
                    exponent_bits,
                };
                format.validate().unwrap();
                let sign = 1u32 << (bits - 1);
                let exponent_max = (1u32 << exponent_bits) - 1;
                let mantissa_max = (1u32 << mantissa_bits) - 1;
                let bias = (1u32 << (exponent_bits - 1)) - 1;
                let words = [
                    0,
                    sign,
                    1,
                    mantissa_max,
                    1 << mantissa_bits,
                    bias << mantissa_bits,
                    (exponent_max - 1) << mantissa_bits | mantissa_max,
                    exponent_max << mantissa_bits,
                    exponent_max << mantissa_bits | 1,
                ];
                let floats = decode(&words, format);
                let mut raw = vec![0; words.len()];
                samples_to_int(&words, &mut raw, format).unwrap();
                let mut restored = vec![0; words.len()];
                int_to_samples(&raw, &mut restored, format).unwrap();
                assert_eq!(restored, words);
                for (&word, &value) in words.iter().zip(&floats) {
                    let exponent = (word & (sign - 1)) >> mantissa_bits;
                    let mantissa = word & mantissa_max;
                    if exponent != exponent_max {
                        let magnitude = if exponent == 0 {
                            f64::from(mantissa)
                                * 2.0f64.powi(1 - bias as i32 - i32::from(mantissa_bits))
                        } else {
                            (1.0 + f64::from(mantissa) / f64::from(1u32 << mantissa_bits))
                                * 2.0f64.powi(exponent as i32 - bias as i32)
                        };
                        let expected = (magnitude as f32).to_bits()
                            | if word & sign != 0 { 1 << 31 } else { 0 };
                        assert_eq!(value.to_bits(), expected, "{format:?}, {word:x}");
                    }
                    // exp=8 custom subnormals expose the source encoder's
                    // implicit-one behavior; test that separately below.
                    if exponent_bits != 8 || exponent != 0 || mantissa == 0 || bits == 32 {
                        let mut encoded = [0];
                        float_to_int_lossless(&[value], &mut encoded, format).unwrap();
                        assert_eq!(encoded[0] as u32, word);
                    }
                }
            }
        }
    }

    #[test]
    fn source_nan_truncation_and_subnormal_precision_policy_are_explicit() {
        let half = SampleFormat::BINARY16;
        assert_eq!(
            source_encode(&[0x7f80_0001, 0xff80_0001, 0x7fc0_1234], half).unwrap(),
            [0x7c00, 0xfc00, 0x7e00]
        );
        let minimum_half = decode(&[1], half)[0];
        let just_above = f32::from_bits(minimum_half.to_bits() + 1);
        assert_eq!(source_encode(&[just_above.to_bits()], half).unwrap(), [1]);
        for value in [
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0x7fc0_1234),
            just_above,
        ] {
            let mut out = [12345];
            assert!(float_to_int_lossless(&[value], &mut out, half).is_err());
            assert_eq!(out, [12345]);
        }
        // A 16-bit float with an 8-bit exponent has seven mantissa bits.
        // Source decoding maps word 64 to binary32 subnormal 0x00400000;
        // source encoding instead yields word 96 (implicit-one insertion).
        let custom = SampleFormat::Float {
            bits: 16,
            exponent_bits: 8,
        };
        assert_eq!(decode(&[64], custom)[0].to_bits(), 0x0040_0000);
        assert_eq!(source_encode(&[0x0040_0000], custom).unwrap(), [96]);
        assert!(float_to_int_lossless(&[f32::from_bits(0x0040_0000)], &mut [0], custom).is_err());
        for bits in [1, (127 + 16) << 23, 1.0f32.to_bits() + 1] {
            assert!(source_encode(&[bits], half).is_err());
        }
    }

    #[test]
    fn integer_rounding_precision_split_and_u31_raw_mapping() {
        let values = [-2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5];
        for bits in [1, 8, 16, 22, 23, 31] {
            let format = SampleFormat::Unsigned { bits };
            let mut out = [0; 8];
            float_to_int(&values, &mut out, format, 1.0).unwrap();
            assert_eq!(out, [-3, -2, -1, 0, 0, 1, 2, 3]);
        }
        let mut out = [0];
        // At 2^23 + 1 the f32 +0.5 operation rounds to the even integer;
        // the f64 source profile truncates the exact half after adjustment.
        float_to_int(
            &[8_388_609.0],
            &mut out,
            SampleFormat::Unsigned { bits: 22 },
            1.0,
        )
        .unwrap();
        assert_eq!(out, [8_388_610]);
        float_to_int(
            &[8_388_609.0],
            &mut out,
            SampleFormat::Unsigned { bits: 23 },
            1.0,
        )
        .unwrap();
        assert_eq!(out, [8_388_609]);
        let format = SampleFormat::Unsigned { bits: 31 };
        float_to_int(&[1.0], &mut out, format, f64::from(i32::MAX)).unwrap();
        assert_eq!(out, [i32::MAX]);
        float_to_int(&[-1.0], &mut out, format, 2_147_483_648.0).unwrap();
        assert_eq!(out, [i32::MIN]);
        let words = [0, 1, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, i32::MAX as u32];
        let mut encoded = [0; 6];
        let mut restored = [0; 6];
        samples_to_int(&words, &mut encoded, format).unwrap();
        int_to_samples(&encoded, &mut restored, format).unwrap();
        assert_eq!(restored, words);
        assert_ne!(((1u32 << 24) + 1) as f32 as u32, (1 << 24) + 1);
        let inverse_factor = 1.0 + 2.0f64.powi(-24);
        let mut inverse = [0.0];
        int_to_float(
            &[3],
            &mut inverse,
            SampleFormat::Unsigned { bits: 22 },
            inverse_factor,
        )
        .unwrap();
        assert_eq!(inverse[0].to_bits(), 3.0f32.to_bits());
        int_to_float(
            &[3],
            &mut inverse,
            SampleFormat::Unsigned { bits: 23 },
            inverse_factor,
        )
        .unwrap();
        assert_eq!(inverse[0].to_bits(), 3.0f32.to_bits() + 1);
    }

    #[test]
    fn full_signed_words_fit_tokens_but_predictor_residuals_can_exceed_i32() {
        use crate::entropy::HybridUintConfig;
        use crate::modular::{predict_simd, predictor::Predictor, tree::pack_signed};

        let words = [
            0x8000_0000u32,
            0x7fff_ffff,
            0,
            0x3f80_0000,
            0xbf80_0000,
            0xff80_0001,
        ];
        let mut samples = [0; 6];
        samples_to_int(&words, &mut samples, SampleFormat::BINARY32).unwrap();
        for sample in samples {
            let packed = pack_signed(sample);
            let encoded = HybridUintConfig::default().encode(packed);
            assert!(encoded.token <= 127);
            assert!(encoded.nbits <= 29);
            assert_eq!(((packed >> 1) as i32) ^ -((packed & 1) as i32), sample);
        }
        let maximum = HybridUintConfig::default().encode(pack_signed(i32::MIN));
        assert_eq!(
            (maximum.token, maximum.nbits, maximum.bits),
            (127, 29, 0x1fff_ffff)
        );
        let mut residuals = [0; 2];
        predict_simd::residual_row(&samples[..2], 2, 2, 0, Predictor::Gradient, &mut residuals)
            .unwrap();
        assert_eq!(residuals, [i64::from(i32::MIN), 4_294_967_295]);
        assert!(i32::try_from(residuals[1]).is_err());
    }

    #[test]
    fn malformed_formats_rows_samples_and_undefined_casts_are_errors() {
        for format in [
            SampleFormat::Unsigned { bits: 0 },
            SampleFormat::Unsigned { bits: 32 },
            SampleFormat::Float {
                bits: 16,
                exponent_bits: 1,
            },
            SampleFormat::Float {
                bits: 16,
                exponent_bits: 9,
            },
            SampleFormat::Float {
                bits: 9,
                exponent_bits: 8,
            },
            SampleFormat::Float {
                bits: 33,
                exponent_bits: 8,
            },
            SampleFormat::Float {
                bits: 255,
                exponent_bits: 255,
            },
        ] {
            let mut out = [7];
            assert!(float_to_int(&[1.0], &mut out, format, 1.0).is_err());
            assert_eq!(out, [7]);
            assert!(samples_to_int(&[0u32], &mut out, format).is_err());
            assert_eq!(out, [7]);
        }
        let format = SampleFormat::Unsigned { bits: 31 };
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2_147_483_648.0] {
            let mut out = [7];
            assert!(float_to_int(&[value], &mut out, format, 1.0).is_err());
            assert_eq!(out, [7]);
        }
        let mut out = [7];
        assert!(float_to_int(&[1.0], &mut out, format, f64::NAN).is_err());
        assert!(float_to_int(&[], &mut out, format, 1.0).is_err());
        assert!(float_to_int_lossless(&[1.0], &mut out, format).is_err());
        assert!(samples_to_int(&[0x8000_0000u32], &mut out, format).is_err());
        assert_eq!(out, [7]);
        assert!(int_to_samples(&[-1], &mut [7], format).is_err());
        assert!(int_to_float(&[-1], &mut [0.0], SampleFormat::BINARY16, 1.0).is_err());
        let values = [1.0, f32::INFINITY, 2.0];
        let mut out = [7; 3];
        assert!(float_to_int(&values, &mut out, format, 1.0).is_err());
        assert_eq!(out, [1, 7, 7]);
        float_to_int(&[], &mut [], format, 1.0).unwrap();
        int_to_float(&[], &mut [], SampleFormat::BINARY32, f64::NAN).unwrap();
    }
}
