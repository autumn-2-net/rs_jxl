// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Original-color ImageMetadata wire port from image_metadata.{h,cc},
//! headers.cc (SizeHeader), encode.cc (intrinsic basic-info policy),
//! frame_header.h (VisitNameString), fields.h and enc_fields.cc at b866b3d.
//! No XYB, preview, extensions, or custom upsampling kernels.
//! Integer 32-bit is rejected by this source revision
//! (BitDepth::VisitFields), despite its header comment saying 1..32.
//! Coded dimensions, frame headers, pixel layouts and byte padding belong to caller.

#![forbid(unsafe_code)]

use crate::bit_writer::BitWriter;
use crate::codestream::write_choice;
use crate::color::{ColorEncoding, write_enum};
use crate::icc::{IccColorSpace, validate_profile};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BitDepth {
    pub bits_per_sample: u8,
    /// Zero for unsigned integer samples, 2..8 for a floating-point exponent.
    pub exponent_bits_per_sample: u8,
}

/// Source AnimationHeader. A zero loop count means unlimited repetition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Animation {
    pub ticks_per_second_numerator: u32,
    pub ticks_per_second_denominator: u32,
    pub loops: u32,
    pub timecodes: bool,
}
impl Default for Animation {
    fn default() -> Self {
        Self {
            ticks_per_second_numerator: 1000,
            ticks_per_second_denominator: 1,
            loops: 0,
            timecodes: false,
        }
    }
}
impl Animation {
    pub fn validate(self) -> Result<()> {
        if self.ticks_per_second_numerator == 0
            || self.ticks_per_second_numerator > 1 << 30
            || self.ticks_per_second_denominator == 0
            || self.ticks_per_second_denominator > 1024
        {
            return Err(Error::InvalidInput(
                "animation timebase exceeds wire range".into(),
            ));
        }
        Ok(())
    }
    fn write_header(self, w: &mut BitWriter) -> Result<()> {
        self.validate()?;
        write_choice(
            w,
            self.ticks_per_second_numerator,
            [(100, 0), (1000, 0), (1, 10), (1, 30)],
        )?;
        write_choice(
            w,
            self.ticks_per_second_denominator,
            [(1, 0), (1001, 0), (1, 8), (1, 10)],
        )?;
        write_choice(w, self.loops, [(0, 0), (0, 3), (0, 16), (0, 32)])?;
        w.write(1, u64::from(self.timecodes))?;
        Ok(())
    }
}

impl BitDepth {
    pub fn new(bits_per_sample: u8) -> Result<Self> {
        let result = Self {
            bits_per_sample,
            exponent_bits_per_sample: 0,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(self) -> Result<()> {
        if self.is_float() {
            let mantissa =
                i16::from(self.bits_per_sample) - i16::from(self.exponent_bits_per_sample) - 1;
            if !(2..=8).contains(&self.exponent_bits_per_sample) || !(2..=23).contains(&mantissa) {
                return Err(Error::InvalidInput(
                    "float depth requires 2..8 exponent and 2..23 mantissa bits".into(),
                ));
            }
        } else if !(1..=31).contains(&self.bits_per_sample) {
            return Err(Error::InvalidInput(
                "integer bit depth must be 1..31 (libjxl b866b3d rejects 32)".into(),
            ));
        }
        Ok(())
    }

    pub fn write_header(self, writer: &mut BitWriter) -> Result<()> {
        self.validate()?;
        writer.write(1, u64::from(self.is_float()))?;
        if self.is_float() {
            write_choice(
                writer,
                u32::from(self.bits_per_sample),
                [(32, 0), (16, 0), (24, 0), (1, 6)],
            )?;
            writer.write(4, u64::from(self.exponent_bits_per_sample - 1))?;
            return Ok(());
        }
        write_choice(
            writer,
            u32::from(self.bits_per_sample),
            [(8, 0), (10, 0), (12, 0), (1, 6)],
        )
    }

    pub const fn is_float(self) -> bool {
        self.exponent_bits_per_sample != 0
    }

    pub fn floating(bits_per_sample: u8, exponent_bits_per_sample: u8) -> Result<Self> {
        let depth = Self {
            bits_per_sample,
            exponent_bits_per_sample,
        };
        if !depth.is_float() {
            return Err(Error::InvalidInput(
                "floating samples require exponent bits".into(),
            ));
        }
        depth.validate()?;
        Ok(depth)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum Orientation {
    #[default]
    Identity = 1,
    FlipHorizontal = 2,
    Rotate180 = 3,
    FlipVertical = 4,
    Transpose = 5,
    Rotate90Clockwise = 6,
    AntiTranspose = 7,
    Rotate90CounterClockwise = 8,
}

impl TryFrom<u8> for Orientation {
    type Error = Error;
    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Identity),
            2 => Ok(Self::FlipHorizontal),
            3 => Ok(Self::Rotate180),
            4 => Ok(Self::FlipVertical),
            5 => Ok(Self::Transpose),
            6 => Ok(Self::Rotate90Clockwise),
            7 => Ok(Self::AntiTranspose),
            8 => Ok(Self::Rotate90CounterClockwise),
            _ => Err(Error::InvalidInput("orientation must be 1..8".into())),
        }
    }
}

/// Suggested display/resampling size, independent of coded pixels and orientation.
/// Both dimensions use the source encoder API range 1..=2^30. A missing value in
/// ImageMetadata means no intrinsic-size field, not a zero-sized display image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntrinsicSize {
    width: u32,
    height: u32,
}

impl IntrinsicSize {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let size = Self { width, height };
        size.validate()?;
        Ok(size)
    }

    pub const fn dimensions(self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn validate(self) -> Result<()> {
        if !(1..=1 << 30).contains(&self.width) || !(1..=1 << 30).contains(&self.height) {
            return Err(Error::InvalidInput(
                "intrinsic dimensions must both be 1..=2^30".into(),
            ));
        }
        Ok(())
    }

    /// Source SizeHeader::Set/VisitFields, without alignment or image allocation.
    /// Semantic validation happens before the first write. An explicit intrinsic
    /// size equal to the coded size is legal; set_intrinsic_size omits that case.
    pub fn write_header(self, writer: &mut BitWriter) -> Result<()> {
        self.validate()?;
        // FindAspectRatio uses the FIRST match and truncates the rational product.
        const RATIOS: [(u32, u32); 7] = [(1, 1), (12, 10), (4, 3), (3, 2), (16, 9), (5, 4), (2, 1)];
        let ratio = RATIOS
            .iter()
            .position(|&(n, d)| {
                u64::from(self.height) * u64::from(n) / u64::from(d) == u64::from(self.width)
            })
            .map_or(0, |i| i as u8 + 1);
        let small = self.height <= 256
            && self.height.is_multiple_of(8)
            && (ratio != 0 || (self.width <= 256 && self.width.is_multiple_of(8)));
        writer.write(1, u64::from(small))?;
        let choices = [(1, 9), (1, 13), (1, 18), (1, 30)];
        if small {
            writer.write(5, u64::from(self.height / 8 - 1))?;
        } else {
            write_choice(writer, self.height, choices)?;
        }
        writer.write(3, u64::from(ratio))?;
        if ratio == 0 {
            if small {
                writer.write(5, u64::from(self.width / 8 - 1))?;
            } else {
                write_choice(writer, self.width, choices)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ExtraChannelType {
    Alpha = 0,
    Depth = 1,
    SpotColor = 2,
    SelectionMask = 3,
    Black = 4,
    Cfa = 5,
    Thermal = 6,
    Optional = 16,
}

impl TryFrom<u32> for ExtraChannelType {
    type Error = Error;
    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Alpha),
            1 => Ok(Self::Depth),
            2 => Ok(Self::SpotColor),
            3 => Ok(Self::SelectionMask),
            4 => Ok(Self::Black),
            5 => Ok(Self::Cfa),
            6 => Ok(Self::Thermal),
            16 => Ok(Self::Optional),
            _ => Err(Error::InvalidInput(
                "unknown or reserved extra channel type".into(),
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExtraChannelInfo {
    pub channel_type: ExtraChannelType,
    pub bit_depth: BitDepth,
    /// Extra dimensions are ceil(color dimension / 2^dim_shift), shift 0..3.
    pub dim_shift: u8,
    /// UTF-8, at most 1071 encoded bytes. Embedded NUL bytes are preserved.
    pub name: String,
    pub alpha_associated: bool,
    /// Linear RGBA; encoded with the source's truncating F16 conversion.
    pub spot_color: [f32; 4],
    pub cfa_channel: u32,
}

impl ExtraChannelInfo {
    pub fn new(channel_type: ExtraChannelType, bits: u8) -> Result<Self> {
        Ok(Self {
            channel_type,
            bit_depth: BitDepth::new(bits)?,
            dim_shift: 0,
            name: String::new(),
            alpha_associated: false,
            spot_color: [0.0; 4],
            cfa_channel: 1,
        })
    }

    pub fn validate(&self) -> Result<()> {
        self.bit_depth.validate()?;
        if self.dim_shift > 3 || self.name.len() > 1071 {
            return Err(Error::InvalidInput(
                "extra channel shift or name exceeds wire range".into(),
            ));
        }
        // Do not silently discard conditional fields that have no representation.
        if self.channel_type != ExtraChannelType::Alpha && self.alpha_associated {
            return Err(Error::InvalidInput(
                "association is only defined for alpha".into(),
            ));
        }
        if self.channel_type != ExtraChannelType::SpotColor && self.spot_color != [0.0; 4] {
            return Err(Error::InvalidInput(
                "spot RGBA requires a spot-color extra channel".into(),
            ));
        }
        if self.channel_type != ExtraChannelType::Cfa && self.cfa_channel != 1 {
            return Err(Error::InvalidInput(
                "CFA index requires a CFA extra channel".into(),
            ));
        }
        for c in self.spot_color {
            f16_bits(c)?;
        }
        if self.cfa_channel > 274 {
            return Err(Error::InvalidInput("CFA index must be 0..274".into()));
        }
        Ok(())
    }

    pub fn write_header(&self, writer: &mut BitWriter) -> Result<()> {
        self.validate()?;
        let default = self.channel_type == ExtraChannelType::Alpha
            && self.bit_depth.bits_per_sample == 8
            && !self.bit_depth.is_float()
            && self.dim_shift == 0
            && self.name.is_empty()
            && !self.alpha_associated;
        writer.write(1, u64::from(default))?;
        if default {
            return Ok(());
        }
        write_enum(writer, self.channel_type as u32)?;
        self.bit_depth.write_header(writer)?;
        write_choice(
            writer,
            u32::from(self.dim_shift),
            [(0, 0), (3, 0), (4, 0), (1, 3)],
        )?;
        write_choice(
            writer,
            self.name.len() as u32,
            [(0, 0), (0, 4), (16, 5), (48, 10)],
        )?;
        for &byte in self.name.as_bytes() {
            writer.write(8, u64::from(byte))?;
        }
        match self.channel_type {
            ExtraChannelType::Alpha => writer.write(1, u64::from(self.alpha_associated))?,
            ExtraChannelType::SpotColor => {
                for c in self.spot_color {
                    writer.write(16, u64::from(f16_bits(c)?))?;
                }
            }
            ExtraChannelType::Cfa => {
                write_choice(writer, self.cfa_channel, [(1, 0), (0, 2), (3, 4), (19, 8)])?;
            }
            _ => (),
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneMapping {
    pub intensity_target: f32,
    pub min_nits: f32,
    pub relative_to_max_display: bool,
    pub linear_below: f32,
}

impl Default for ToneMapping {
    fn default() -> Self {
        Self {
            intensity_target: 255.0,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
        }
    }
}

impl ToneMapping {
    pub fn validate(self) -> Result<()> {
        let intensity = f16_bits(self.intensity_target)?;
        f16_bits(self.min_nits)?;
        f16_bits(self.linear_below)?;
        if self.intensity_target <= 0.0
            || intensity == 0
            || self.min_nits < 0.0
            || self.min_nits > self.intensity_target
            || self.linear_below < 0.0
            || (self.relative_to_max_display && self.linear_below > 1.0)
        {
            return Err(Error::InvalidInput(
                "invalid tone mapping or F16-underflow intensity".into(),
            ));
        }
        Ok(())
    }

    pub fn write_header(self, writer: &mut BitWriter) -> Result<()> {
        self.validate()?;
        let default = self == Self::default();
        writer.write(1, u64::from(default))?;
        if !default {
            writer.write(16, u64::from(f16_bits(self.intensity_target)?))?;
            writer.write(16, u64::from(f16_bits(self.min_nits)?))?;
            writer.write(1, u64::from(self.relative_to_max_display))?;
            writer.write(16, u64::from(f16_bits(self.linear_below)?))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageMetadata {
    pub animation: Option<Animation>,
    pub bit_depth: BitDepth,
    pub orientation: Orientation,
    /// Display size only; does not change the coded canvas, samples or allocation.
    pub intrinsic_size: Option<IntrinsicSize>,
    pub extra_channels: Vec<ExtraChannelInfo>,
    pub color_encoding: ColorEncoding,
    pub tone_mapping: ToneMapping,
    /// Caller promises every modular intermediate fits 16 bits if true.
    /// Defaults false; sample depth alone cannot establish this after transforms.
    pub modular_16_bit_buffer_sufficient: bool,
}

impl ImageMetadata {
    /// CMYK callers must add a Black extra channel before validate/write_header.
    pub fn new(bits: u8, color_encoding: ColorEncoding) -> Result<Self> {
        color_encoding.validate()?;
        let result = Self {
            animation: None,
            bit_depth: BitDepth::new(bits)?,
            orientation: Orientation::Identity,
            intrinsic_size: None,
            extra_channels: Vec::new(),
            color_encoding,
            tone_mapping: ToneMapping::default(),
            modular_16_bit_buffer_sufficient: false,
        };
        Ok(result)
    }

    /// Applies JxlEncoderSetBasicInfo's intrinsic-size omission policy to a fresh
    /// value: (0, 0) or the coded size clears it; one zero or a dimension above
    /// 2^30 is rejected. `coded_size` must describe the already-validated canvas.
    /// Values are not swapped for orientation and errors leave metadata unchanged.
    pub fn set_intrinsic_size(
        &mut self,
        coded_size: (u32, u32),
        intrinsic_size: (u32, u32),
    ) -> Result<()> {
        let value = if intrinsic_size == (0, 0) || intrinsic_size == coded_size {
            None
        } else {
            Some(IntrinsicSize::new(intrinsic_size.0, intrinsic_size.1)?)
        };
        self.intrinsic_size = value;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if let Some(animation) = self.animation {
            animation.validate()?;
        }
        if let Some(intrinsic_size) = self.intrinsic_size {
            intrinsic_size.validate()?;
        }
        self.bit_depth.validate()?;
        self.color_encoding.validate()?;
        self.tone_mapping.validate()?;
        if self.extra_channels.len() > 4096 {
            return Err(Error::Limit("too many extra channels"));
        }
        for extra in &self.extra_channels {
            extra.validate()?;
        }
        let black = self
            .extra_channels
            .iter()
            .filter(|e| e.channel_type == ExtraChannelType::Black)
            .count();
        let cmyk = match self.color_encoding.icc_profile() {
            Some(profile) => validate_profile(profile)? == IccColorSpace::Cmyk,
            None => false,
        };
        if black != usize::from(cmyk) {
            return Err(Error::InvalidInput(
                "CMYK ICC requires exactly one Black channel; other color encodings require none"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Writes ImageMetadata only (xyb_encoded=false). Call after SizeHeader,
    /// then write_default_transform_data, optional icc::write_icc, then pad.
    /// All semantic validation happens before the first write.
    pub fn write_header(&self, writer: &mut BitWriter) -> Result<()> {
        self.validate()?;
        writer.write(1, 0)?; // Original color is never the XYB metadata default.
        let extra_fields = self.orientation != Orientation::Identity
            || self.tone_mapping != ToneMapping::default()
            || self.animation.is_some()
            || self.intrinsic_size.is_some();
        writer.write(1, u64::from(extra_fields))?;
        if extra_fields {
            writer.write(3, u64::from(self.orientation as u8 - 1))?;
            writer.write(1, u64::from(self.intrinsic_size.is_some()))?;
            if let Some(intrinsic_size) = self.intrinsic_size {
                intrinsic_size.write_header(writer)?;
            }
            writer.write(1, 0)?; // no preview
            writer.write(1, u64::from(self.animation.is_some()))?;
            if let Some(animation) = self.animation {
                animation.write_header(writer)?;
            }
        }
        self.bit_depth.write_header(writer)?;
        writer.write(1, u64::from(self.modular_16_bit_buffer_sufficient))?;
        write_choice(
            writer,
            self.extra_channels.len() as u32,
            [(0, 0), (1, 0), (2, 4), (1, 12)],
        )?;
        for extra in &self.extra_channels {
            extra.write_header(writer)?;
        }
        writer.write(1, 0)?; // xyb_encoded
        self.color_encoding.write_header(writer)?;
        if extra_fields {
            self.tone_mapping.write_header(writer)?;
        }
        writer.write(2, 0)?; // U64 zero: no extensions
        Ok(())
    }
}

pub fn write_default_transform_data(writer: &mut BitWriter) -> Result<()> {
    writer.write(1, 1)?;
    Ok(())
}

// enc_fields.cc F16Coder::Write truncates; Rust/IEEE round-to-nearest differs.
pub(crate) fn f16_bits(value: f32) -> Result<u16> {
    if !value.is_finite() || value.abs() > 65504.0 {
        return Err(Error::InvalidInput(
            "metadata float must be finite and fit F16".into(),
        ));
    }
    f16_bits_truncating(value)
}

// Direct F16Coder::Write callers do not run CanEncode first. Values in
// (65504, 65536) still have exponent 15 and truncate to the largest finite F16.
// Keep the stricter metadata validation above unchanged.
pub(crate) fn f16_bits_truncating(value: f32) -> Result<u16> {
    if !value.is_finite() || value.abs() >= 65536.0 {
        return Err(Error::InvalidInput(
            "float exceeds the source F16 writer range".into(),
        ));
    }
    let bits = value.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 255) as i32 - 127;
    let mantissa = bits & 0x7f_ffff;
    if exp < -24 {
        return Ok(0);
    }
    let (exp16, mantissa16) = if exp < -14 {
        let shift = (-14 - exp) as u32;
        (0, (1 << (10 - shift)) + (mantissa >> (13 + shift)))
    } else {
        ((exp + 15) as u32, mantissa >> 13)
    };
    Ok((sign << 15) | ((exp16 as u16) << 10) | mantissa16 as u16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ColorSpace;

    struct Reader {
        bytes: Vec<u8>,
        pos: usize,
    }
    impl Reader {
        fn new(mut w: BitWriter) -> Self {
            w.zero_pad_to_byte().unwrap();
            Self {
                bytes: w.into_bytes().unwrap(),
                pos: 0,
            }
        }
        fn bits(&mut self, n: usize) -> u32 {
            let mut v = 0;
            for i in 0..n {
                v |= u32::from((self.bytes[self.pos / 8] >> (self.pos % 8)) & 1) << i;
                self.pos += 1;
            }
            v
        }
        fn choice(&mut self, offsets: [u32; 4], widths: [usize; 4]) -> u32 {
            let selector = self.bits(2) as usize;
            offsets[selector] + self.bits(widths[selector])
        }
        fn depth(&mut self) -> u32 {
            assert_eq!(self.bits(1), 0);
            self.choice([8, 10, 12, 1], [0, 0, 0, 6])
        }
        fn size(&mut self) -> (u32, u32) {
            let small = self.bits(1) != 0;
            let height = if small {
                8 * (self.bits(5) + 1)
            } else {
                self.choice([1; 4], [9, 13, 18, 30])
            };
            let ratio = self.bits(3);
            let width = match ratio {
                0 if small => 8 * (self.bits(5) + 1),
                0 => self.choice([1; 4], [9, 13, 18, 30]),
                1 => height,
                2 => (u64::from(height) * 12 / 10) as u32,
                3 => (u64::from(height) * 4 / 3) as u32,
                4 => (u64::from(height) * 3 / 2) as u32,
                5 => (u64::from(height) * 16 / 9) as u32,
                6 => (u64::from(height) * 5 / 4) as u32,
                7 => height * 2,
                _ => unreachable!(),
            };
            (width, height)
        }
    }
    fn rgb(bits: u8) -> ImageMetadata {
        ImageMetadata::new(bits, ColorEncoding::srgb(ColorSpace::Rgb)).unwrap()
    }

    #[test]
    fn source_wire_goldens_rgb_and_default_alpha() {
        // Bit positions hand-derived from ImageMetadata::VisitFields, not a
        // roundtrip through this serializer. 16-bit-buffer promise stays false.
        let mut m = rgb(8);
        for (alpha, expected, bits) in [(false, [0x00, 0x02], 12), (true, [0x40, 0x05], 13)] {
            if alpha {
                m.extra_channels
                    .push(ExtraChannelInfo::new(ExtraChannelType::Alpha, 8).unwrap());
            }
            let mut w = BitWriter::new(8);
            m.write_header(&mut w).unwrap();
            assert_eq!(w.bits_written(), bits);
            assert_eq!(Reader::new(w).bytes, expected);
        }
    }

    #[test]
    fn intrinsic_size_source_wire_goldens_and_all_seven_aspect_ratios() {
        for (width, height, bits, expected) in [
            (1, 1, 15, vec![0x00, 0x10]),
            (8, 8, 9, vec![0x41, 0x00]),
            (9, 8, 9, vec![0x81, 0x00]), // Truncated 12/10; width need not divide by 8.
            (16, 8, 9, vec![0xc1, 0x01]),
            (512, 256, 9, vec![0xff, 0x01]), // Small flag despite width > 256.
            (64, 256, 14, vec![0x3f, 0x0e]),
            (7, 8, 26, vec![0x38, 0x00, 0x0c, 0x00]),
        ] {
            let mut w = BitWriter::new(16);
            IntrinsicSize::new(width, height)
                .unwrap()
                .write_header(&mut w)
                .unwrap();
            assert_eq!(w.bits_written(), bits);
            let mut r = Reader::new(w);
            assert_eq!(r.bytes, expected, "{width}x{height}");
            assert_eq!(r.size(), (width, height));
            assert_eq!(r.pos, bits);
        }
        for (index, width) in [360, 432, 480, 540, 640, 450, 720].into_iter().enumerate() {
            let mut w = BitWriter::new(16);
            IntrinsicSize::new(width, 360)
                .unwrap()
                .write_header(&mut w)
                .unwrap();
            let mut r = Reader::new(w);
            assert_eq!(r.bits(1), 0);
            assert_eq!(r.bits(2), 0);
            assert_eq!(r.bits(9), 359);
            assert_eq!(r.bits(3), index as u32 + 1);
            assert_eq!(r.pos, 15);
        }
    }

    #[test]
    fn intrinsic_size_explicit_dimensions_cover_all_u32_selector_boundaries() {
        for (dimension, selector, width) in [
            (3, 0, 9),
            (511, 0, 9),
            (512, 0, 9),
            (513, 1, 13),
            (8192, 1, 13),
            (8193, 2, 18),
            (262144, 2, 18),
            (262145, 3, 30),
            (1 << 30, 3, 30),
        ] {
            for portrait in [false, true] {
                let dimensions = if portrait {
                    (1, dimension)
                } else {
                    (dimension, 1)
                };
                let size = IntrinsicSize::new(dimensions.0, dimensions.1).unwrap();
                let mut w = BitWriter::new(16);
                size.write_header(&mut w).unwrap();
                let mut r = Reader::new(w);
                assert_eq!(r.bits(1), 0);
                assert_eq!(r.bits(2), if portrait { selector } else { 0 });
                assert_eq!(
                    r.bits(if portrait { width } else { 9 }),
                    if portrait { dimension - 1 } else { 0 }
                );
                assert_eq!(r.bits(3), 0); // None of these portrait/landscape ratios is implicit.
                assert_eq!(r.bits(2), if portrait { 0 } else { selector });
                assert_eq!(
                    r.bits(if portrait { 9 } else { width }),
                    if portrait { 0 } else { dimension - 1 }
                );
                r.pos = 0;
                assert_eq!(r.size(), dimensions);
                assert_eq!(size.dimensions(), dimensions);
            }
        }
        // Many ratios collide at small sizes. FindAspectRatio must use the first.
        let mut w = BitWriter::new(16);
        IntrinsicSize::new(1, 1)
            .unwrap()
            .write_header(&mut w)
            .unwrap();
        let mut r = Reader::new(w);
        r.bits(12);
        assert_eq!(r.bits(3), 1);
    }

    #[test]
    fn intrinsic_basic_info_policy_preserves_defaults_and_errors_are_atomic() {
        let mut m = rgb(8);
        assert_eq!(m.intrinsic_size, None);
        for input in [(0, 0), (19, 7)] {
            m.intrinsic_size = Some(IntrinsicSize::new(30, 20).unwrap());
            m.set_intrinsic_size((19, 7), input).unwrap();
            assert_eq!(m, rgb(8));
            let mut w = BitWriter::new(8);
            m.write_header(&mut w).unwrap();
            assert_eq!(Reader::new(w).bytes, [0, 2]);
        }
        m.set_intrinsic_size((19, 7), (1 << 30, 1)).unwrap();
        assert_eq!(m.intrinsic_size.unwrap().dimensions(), (1 << 30, 1));
        let before = m.clone();
        for dimensions in [(0, 1), (1, 0), ((1 << 30) + 1, 1), (1, u32::MAX)] {
            assert!(IntrinsicSize::new(dimensions.0, dimensions.1).is_err());
            assert!(m.set_intrinsic_size((19, 7), dimensions).is_err());
            assert_eq!(m, before);
        }
        assert!(IntrinsicSize::new(0, 0).is_err());
        // Direct explicit metadata is legal even if a caller could omit it.
        m.intrinsic_size = Some(IntrinsicSize::new(19, 7).unwrap());
        m.validate().unwrap();
    }

    #[test]
    fn intrinsic_metadata_field_order_coexists_with_orientation_animation_and_tone() {
        let mut m = rgb(8);
        m.intrinsic_size = Some(IntrinsicSize::new(8, 8).unwrap());
        let mut w = BitWriter::new(16);
        m.write_header(&mut w).unwrap();
        assert_eq!(w.bits_written(), 28);
        assert_eq!(Reader::new(w).bytes, [0x62, 0x10, 0x00, 0x03]);
        for orientation in 1..=8 {
            for animated in [false, true] {
                let mut m = rgb(8);
                m.orientation = Orientation::try_from(orientation).unwrap();
                m.intrinsic_size = Some(IntrinsicSize::new(9, 8).unwrap());
                m.animation = animated.then(Animation::default);
                m.tone_mapping.intensity_target = 1000.0;
                let mut w = BitWriter::new(32);
                m.write_header(&mut w).unwrap();
                let written = w.bits_written();
                let mut r = Reader::new(w);
                assert_eq!(r.bits(1), 0); // Nondefault metadata.
                assert_eq!(r.bits(1), 1); // Extra fields.
                assert_eq!(r.bits(3), u32::from(orientation - 1));
                assert_eq!(r.bits(1), 1); // Intrinsic size precedes preview/animation.
                assert_eq!(r.size(), (9, 8)); // Never swapped for orientation 5..8.
                assert_eq!(r.bits(1), 0); // No preview.
                assert_eq!(r.bits(1), u32::from(animated));
                if animated {
                    assert_eq!(r.bits(2), 1); // 1000 ticks/sec.
                    assert_eq!(r.bits(2), 0); // Denominator 1.
                    assert_eq!(r.bits(2), 0); // Infinite looping.
                    assert_eq!(r.bits(1), 0); // No timecodes.
                }
                assert_eq!(r.depth(), 8);
                assert_eq!(r.bits(1), 0); // Sufficient buffer.
                assert_eq!(r.bits(2), 0); // No extras.
                assert_eq!(r.bits(1), 0); // No XYB.
                assert_eq!(r.bits(1), 1); // Default sRGB.
                assert_eq!(r.bits(1), 0); // Explicit tone mapping.
                assert_eq!(r.bits(16), 0x63d0);
                assert_eq!(r.bits(16), 0);
                assert_eq!(r.bits(1), 0);
                assert_eq!(r.bits(16), 0);
                assert_eq!(r.bits(2), 0); // No extensions.
                assert_eq!(r.pos, written);
            }
        }
    }

    #[test]
    fn intrinsic_size_validates_before_writing_and_does_not_align_or_resize_pixels() {
        let bad = IntrinsicSize {
            width: 0,
            height: 8,
        };
        let mut w = BitWriter::new(32);
        w.write(3, 5).unwrap();
        assert!(bad.write_header(&mut w).is_err());
        assert_eq!(w.bits_written(), 3);
        let mut m = rgb(8);
        m.intrinsic_size = Some(bad);
        assert!(m.write_header(&mut w).is_err());
        assert_eq!(w.bits_written(), 3);
        let size = IntrinsicSize::new(9, 8).unwrap();
        size.write_header(&mut w).unwrap();
        assert_eq!(w.bits_written(), 12);
        let mut r = Reader::new(w);
        assert_eq!(r.bits(3), 5);
        assert_eq!(r.size(), (9, 8));
        assert!(size.write_header(&mut BitWriter::new(1)).is_err());

        let pixels = crate::PixelData::U8(vec![17; 2 * 3 * 3]);
        let image = crate::Image::new(2, 3, crate::PixelLayout::Rgb, pixels.clone()).unwrap();
        let mut metadata = image.metadata().clone();
        metadata.intrinsic_size = Some(IntrinsicSize::new(1 << 30, 1 << 30).unwrap());
        let image = image.with_metadata(metadata, Vec::new()).unwrap();
        assert_eq!((image.width(), image.height()), (2, 3));
        assert_eq!(image.pixels(), &pixels);
        let header = crate::codestream::image_header_fields(2, 3, image.metadata(), 128).unwrap();
        let mut r = Reader {
            bytes: header,
            pos: 0,
        };
        assert_eq!(r.bits(16), 0xaff);
        assert_eq!(r.size(), (2, 3)); // The actual image SizeHeader is unchanged.
        assert_eq!(r.bits(1), 0);
        assert_eq!(r.bits(1), 1);
        assert_eq!(r.bits(3), 0);
        assert_eq!(r.bits(1), 1);
        assert_eq!(r.size(), (1 << 30, 1 << 30));
    }

    #[test]
    fn integer_depths_and_orientations_cover_wire_boundaries() {
        for bits in 1..=31 {
            let mut w = BitWriter::new(8);
            BitDepth::new(bits).unwrap().write_header(&mut w).unwrap();
            assert_eq!(Reader::new(w).depth(), u32::from(bits));
        }
        for bits in [0, 32, 64, 255] {
            assert!(BitDepth::new(bits).is_err());
        }
        for value in 1..=8 {
            let mut m = rgb(16);
            m.orientation = Orientation::try_from(value).unwrap();
            let mut w = BitWriter::new(16);
            m.write_header(&mut w).unwrap();
            let mut r = Reader::new(w);
            assert_eq!(r.bits(1), 0);
            assert_eq!(r.bits(1), u32::from(value != 1));
            if value != 1 {
                assert_eq!(r.bits(3), u32::from(value - 1));
                assert_eq!(r.bits(3), 0);
            }
            assert_eq!(r.depth(), 16);
            assert_eq!(r.bits(1), 0); // sufficient buffer
            assert_eq!(r.bits(2), 0); // extras
            assert_eq!(r.bits(1), 0); // XYB
            assert_eq!(r.bits(1), 1); // sRGB default
            if value != 1 {
                assert_eq!(r.bits(1), 1);
            } // tone default
            assert_eq!(r.bits(2), 0);
        }
        assert!(Orientation::try_from(0).is_err());
        assert!(Orientation::try_from(9).is_err());
    }

    #[test]
    fn floating_depths_use_source_choices_and_four_exponent_bits() {
        for exponent in 2..=8 {
            for mantissa in 2..=23 {
                let bits = exponent + mantissa + 1;
                let depth = BitDepth::floating(bits, exponent).unwrap();
                let mut w = BitWriter::new(8);
                depth.write_header(&mut w).unwrap();
                let mut r = Reader::new(w);
                assert_eq!(r.bits(1), 1);
                assert_eq!(r.choice([32, 16, 24, 1], [0, 0, 0, 6]), u32::from(bits));
                assert_eq!(r.bits(4) + 1, u32::from(exponent));
            }
        }
        assert!(BitDepth::floating(32, 7).is_err());
        assert!(BitDepth::floating(16, 0).is_err());
    }

    #[test]
    fn typed_extras_serialize_conditional_fields_and_utf8_names() {
        for id in [0, 1, 2, 3, 4, 5, 6, 16] {
            let mut e = ExtraChannelInfo::new(ExtraChannelType::try_from(id).unwrap(), 16).unwrap();
            e.dim_shift = 3;
            e.name = "x\0\u{4e2d}".into();
            if id == 0 {
                e.alpha_associated = true;
            }
            if id == 2 {
                e.spot_color = [1.0, -0.5, 0.25, 0.75];
            }
            if id == 5 {
                e.cfa_channel = 274;
            }
            let mut w = BitWriter::new(128);
            e.write_header(&mut w).unwrap();
            let count = w.bits_written();
            let mut r = Reader::new(w);
            assert_eq!(r.bits(1), 0);
            assert_eq!(r.choice([0, 1, 2, 18], [0, 0, 4, 6]), id);
            assert_eq!(r.depth(), 16);
            assert_eq!(r.choice([0, 3, 4, 1], [0, 0, 0, 3]), 3);
            assert_eq!(r.choice([0, 0, 16, 48], [0, 4, 5, 10]), e.name.len() as u32);
            for &b in e.name.as_bytes() {
                assert_eq!(r.bits(8), u32::from(b));
            }
            match id {
                0 => assert_eq!(r.bits(1), 1),
                2 => {
                    for expected in [0x3c00, 0xb800, 0x3400, 0x3a00] {
                        assert_eq!(r.bits(16), expected);
                    }
                }
                5 => assert_eq!(r.choice([1, 0, 3, 19], [0, 2, 4, 8]), 274),
                _ => (),
            }
            assert_eq!(r.pos, count);
        }
        for id in 7..=15 {
            assert!(ExtraChannelType::try_from(id).is_err());
        }
        assert!(ExtraChannelType::try_from(17).is_err());
    }

    #[test]
    fn source_f16_truncation_subnormals_and_invalid_values() {
        for (value, expected) in [
            (0.0, 0),
            (-0.0, 0),
            (1.0009, 0x3c00),
            (-1.0009, 0xbc00),
            (65504.0, 0x7bff),
            (2.0f32.powi(-24), 1),
            (-2.0f32.powi(-24), 0x8001),
            (2.0f32.powi(-25), 0),
            (-2.0f32.powi(-25), 0),
        ] {
            assert_eq!(f16_bits(value).unwrap(), expected);
        }
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 65536.0] {
            assert!(f16_bits(v).is_err());
        }
        let mut t = ToneMapping {
            intensity_target: 1000.0,
            min_nits: 0.25,
            relative_to_max_display: true,
            linear_below: 0.5,
        };
        let mut w = BitWriter::new(8);
        t.write_header(&mut w).unwrap();
        assert_eq!(w.bits_written(), 50);
        let mut r = Reader::new(w);
        assert_eq!(r.bits(1), 0);
        assert_eq!(r.bits(16), 0x63d0);
        assert_eq!(r.bits(16), 0x3400);
        assert_eq!(r.bits(1), 1);
        assert_eq!(r.bits(16), 0x3800);
        t.linear_below = 1.1;
        assert!(t.validate().is_err());
        t = ToneMapping {
            intensity_target: 2.0f32.powi(-25),
            ..ToneMapping::default()
        };
        assert!(t.validate().is_err());
    }

    #[test]
    fn counts_names_and_validation_before_writes() {
        let mut m = rgb(8);
        let mut e = ExtraChannelInfo::new(ExtraChannelType::Depth, 31).unwrap();
        for len in [0, 1, 15, 16, 47, 48, 1071] {
            e.name = "x".repeat(len);
            e.write_header(&mut BitWriter::new(2048)).unwrap();
        }
        e.name.push('x');
        let mut w = BitWriter::new(1024);
        w.write(3, 5).unwrap();
        assert!(e.write_header(&mut w).is_err());
        assert_eq!(w.bits_written(), 3);
        e.name.clear();
        e.dim_shift = 4;
        assert!(e.validate().is_err());
        e.dim_shift = 0;
        e.alpha_associated = true;
        assert!(e.validate().is_err());
        let alpha = ExtraChannelInfo::new(ExtraChannelType::Alpha, 8).unwrap();
        for count in [0, 1, 2, 17, 18, 4096] {
            m.extra_channels = vec![alpha.clone(); count];
            let mut w = BitWriter::new(8192);
            m.write_header(&mut w).unwrap();
            let mut r = Reader::new(w);
            assert_eq!(r.bits(6), 0);
            assert_eq!(r.choice([0, 1, 2, 1], [0, 0, 4, 12]), count as u32);
        }
        m.extra_channels.push(alpha);
        assert!(m.write_header(&mut w).is_err());
        assert_eq!(w.bits_written(), 3);
    }

    #[test]
    fn cmyk_is_explicit_and_requires_black_not_rgb_fallback() {
        let mut profile = vec![0u8; 132];
        profile[..4].copy_from_slice(&132u32.to_be_bytes());
        profile[16..24].copy_from_slice(b"CMYKLab ");
        profile[36..40].copy_from_slice(b"acsp");
        let mut m =
            ImageMetadata::new(8, ColorEncoding::icc(ColorSpace::Rgb, profile).unwrap()).unwrap();
        assert!(m.validate().is_err());
        m.extra_channels
            .push(ExtraChannelInfo::new(ExtraChannelType::Black, 8).unwrap());
        m.validate().unwrap();
        m.color_encoding = ColorEncoding::srgb(ColorSpace::Rgb);
        assert!(m.validate().is_err());
    }
}
