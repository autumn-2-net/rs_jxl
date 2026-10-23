// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! ColorEncoding wire fields from color_encoding_internal.{h,cc},
//! cms/color_encoding_cms.h and fields.h at libjxl b866b3d.
//! RGB/gray fields and explicit ICC are distinct choices. No color conversion,
//! implicit sRGB fallback, profile synthesis, or CMS equivalence detection.
//! Custom xy uses signed millionths; Gamma uses the source's 1e-7 OETF units.
//! Unknown/XYB spaces and unknown transfer functions are intentionally absent.
//! ICC validation is structural, not a replacement for a full ICC/CMS parser.

#![forbid(unsafe_code)]

use crate::bit_writer::BitWriter;
use crate::codestream::write_choice;
use crate::icc::{IccColorSpace, validate_profile};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ColorSpace {
    Rgb = 0,
    Gray = 1,
}

impl ColorSpace {
    pub const fn channels(self) -> usize {
        match self {
            Self::Rgb => 3,
            Self::Gray => 1,
        }
    }
}

impl TryFrom<u32> for ColorSpace {
    type Error = Error;
    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Rgb),
            1 => Ok(Self::Gray),
            _ => Err(Error::InvalidInput(
                "only explicit RGB or gray color spaces are supported".into(),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Chromaticity {
    pub x: i32,
    pub y: i32,
}

impl Chromaticity {
    /// Matches cms::Customxy::SetValue's roundf AFTER conversion to f32.
    pub fn from_xy(x: f64, y: f64) -> Result<Self> {
        if !x.is_finite() || !y.is_finite() || x.abs() >= 4.0 || y.abs() >= 4.0 {
            return Err(Error::InvalidInput(
                "nonfinite or out-of-range chromaticity".into(),
            ));
        }
        let xy = Self {
            x: ((x * 1_000_000.0) as f32).round() as i32,
            y: ((y * 1_000_000.0) as f32).round() as i32,
        };
        xy.validate()?;
        Ok(xy)
    }

    pub fn validate(self) -> Result<()> {
        if !(-0x20_0000..=0x1f_ffff).contains(&self.x)
            || !(-0x20_0000..=0x1f_ffff).contains(&self.y)
        {
            return Err(Error::InvalidInput(
                "chromaticity exceeds signed 22-bit wire range".into(),
            ));
        }
        Ok(())
    }

    fn write(self, writer: &mut BitWriter) -> Result<()> {
        for v in [self.x, self.y] {
            let packed = ((v as u32) << 1) ^ ((v >> 31) as u32);
            write_choice(
                writer,
                packed,
                [(0, 19), (524288, 19), (1048576, 20), (2097152, 21)],
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhitePoint {
    D65,
    E,
    Dci,
    Custom(Chromaticity),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Primaries {
    Srgb,
    Bt2100,
    P3,
    Custom {
        red: Chromaticity,
        green: Chromaticity,
        blue: Chromaticity,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferFunction {
    Bt709,
    Linear,
    Srgb,
    Pq,
    Dci,
    Hlg,
    /// OETF exponent * 10,000,000; NOT the inverse/display gamma.
    Gamma(u32),
}

impl TransferFunction {
    pub fn validate(self) -> Result<()> {
        if let Self::Gamma(g) = self
            && (g > 10_000_000 || u64::from(g) * 8192 < 10_000_000)
        {
            return Err(Error::InvalidInput(
                "gamma must be in [1/8192, 1] in 1e-7 units".into(),
            ));
        }
        Ok(())
    }

    fn write(self, writer: &mut BitWriter) -> Result<()> {
        if let Self::Gamma(g) = self {
            writer.write(1, 1)?;
            writer.write(24, u64::from(g))?;
        } else {
            writer.write(1, 0)?;
            let id = match self {
                Self::Bt709 => 1,
                Self::Linear => 8,
                Self::Srgb => 13,
                Self::Pq => 16,
                Self::Dci => 17,
                Self::Hlg => 18,
                Self::Gamma(_) => unreachable!(),
            };
            write_enum(writer, id)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum RenderingIntent {
    Perceptual = 0,
    Relative = 1,
    Saturation = 2,
    Absolute = 3,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColorFields {
    pub color_space: ColorSpace,
    pub white_point: WhitePoint,
    /// Required for RGB and must be None for gray, since gray has no wire primaries.
    pub primaries: Option<Primaries>,
    pub transfer_function: TransferFunction,
    pub rendering_intent: RenderingIntent,
}

impl ColorFields {
    pub fn srgb(color_space: ColorSpace) -> Self {
        Self {
            color_space,
            white_point: WhitePoint::D65,
            primaries: if color_space == ColorSpace::Rgb {
                Some(Primaries::Srgb)
            } else {
                None
            },
            transfer_function: TransferFunction::Srgb,
            rendering_intent: RenderingIntent::Relative,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.primaries.is_some() != (self.color_space == ColorSpace::Rgb) {
            return Err(Error::InvalidInput(
                "RGB requires primaries; gray must omit them".into(),
            ));
        }
        if let WhitePoint::Custom(xy) = self.white_point {
            xy.validate()?;
            if !(0..=1_000_000).contains(&xy.x) || !(1..=1_000_000).contains(&xy.y) {
                return Err(Error::InvalidInput("invalid white point".into()));
            }
        }
        if let Some(Primaries::Custom { red, green, blue }) = self.primaries {
            for xy in [red, green, blue] {
                xy.validate()?;
            }
            // The RGB-to-XYZ primary matrix is singular for collinear xy.
            let det = i64::from(red.x - blue.x) * i64::from(green.y - blue.y)
                - i64::from(green.x - blue.x) * i64::from(red.y - blue.y);
            if det == 0 {
                return Err(Error::InvalidInput("singular custom primaries".into()));
            }
        }
        self.transfer_function.validate()
    }

    fn write(&self, writer: &mut BitWriter) -> Result<()> {
        let default = *self == Self::srgb(ColorSpace::Rgb);
        writer.write(1, u64::from(default))?;
        if default {
            return Ok(());
        }
        writer.write(1, 0)?; // want_icc
        write_enum(writer, self.color_space as u32)?;
        let wp = match self.white_point {
            WhitePoint::D65 => 1,
            WhitePoint::Custom(_) => 2,
            WhitePoint::E => 10,
            WhitePoint::Dci => 11,
        };
        write_enum(writer, wp)?;
        if let WhitePoint::Custom(xy) = self.white_point {
            xy.write(writer)?;
        }
        if let Some(primaries) = self.primaries {
            let id = match primaries {
                Primaries::Srgb => 1,
                Primaries::Custom { .. } => 2,
                Primaries::Bt2100 => 9,
                Primaries::P3 => 11,
            };
            write_enum(writer, id)?;
            if let Primaries::Custom { red, green, blue } = primaries {
                for xy in [red, green, blue] {
                    xy.write(writer)?;
                }
            }
        }
        self.transfer_function.write(writer)?;
        write_enum(writer, self.rendering_intent as u32)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ColorEncoding {
    Fields(ColorFields),
    /// RGB also covers CMYK: ImageMetadata requires exactly one Black extra channel.
    Icc {
        color_space: ColorSpace,
        profile: Vec<u8>,
    },
}

impl ColorEncoding {
    pub fn srgb(color_space: ColorSpace) -> Self {
        Self::Fields(ColorFields::srgb(color_space))
    }

    pub fn linear_srgb(color_space: ColorSpace) -> Self {
        let mut fields = ColorFields::srgb(color_space);
        fields.transfer_function = TransferFunction::Linear;
        Self::Fields(fields)
    }

    pub fn icc(color_space: ColorSpace, profile: Vec<u8>) -> Result<Self> {
        let result = Self::Icc {
            color_space,
            profile,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn color_space(&self) -> ColorSpace {
        match self {
            Self::Fields(f) => f.color_space,
            Self::Icc { color_space, .. } => *color_space,
        }
    }

    pub fn icc_profile(&self) -> Option<&[u8]> {
        match self {
            Self::Fields(_) => None,
            Self::Icc { profile, .. } => Some(profile),
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Fields(fields) => fields.validate(),
            Self::Icc {
                color_space,
                profile,
            } => {
                let icc_space = validate_profile(profile)?;
                if matches!(
                    (*color_space, icc_space),
                    (ColorSpace::Gray, IccColorSpace::Gray)
                        | (ColorSpace::Rgb, IccColorSpace::Rgb | IccColorSpace::Cmyk)
                ) {
                    Ok(())
                } else {
                    Err(Error::InvalidInput(
                        "ICC color space does not match color channels".into(),
                    ))
                }
            }
        }
    }

    /// Writes only the ColorEncoding bundle; ICC bytes belong after custom
    /// transform data, via icc::write_icc, and before byte padding.
    pub fn write_header(&self, writer: &mut BitWriter) -> Result<()> {
        self.validate()?;
        match self {
            Self::Fields(fields) => fields.write(writer),
            Self::Icc { color_space, .. } => {
                writer.write(1, 0)?; // all_default
                writer.write(1, 1)?; // want_icc
                write_enum(writer, *color_space as u32)
            }
        }
    }
}

pub(crate) fn write_enum(writer: &mut BitWriter, value: u32) -> Result<()> {
    write_choice(writer, value, [(0, 0), (1, 0), (2, 4), (18, 6)])
}

#[cfg(test)]
mod tests {
    use super::*;

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
        fn enumeration(&mut self) -> u32 {
            match self.bits(2) {
                0 => 0,
                1 => 1,
                2 => 2 + self.bits(4),
                _ => 18 + self.bits(6),
            }
        }
        fn xy(&mut self) -> Chromaticity {
            let mut values = [0; 2];
            for v in &mut values {
                let selector = self.bits(2) as usize;
                let packed =
                    [0, 524288, 1048576, 2097152][selector] + self.bits([19, 19, 20, 21][selector]);
                *v = (packed >> 1) as i32 ^ -((packed & 1) as i32);
            }
            Chromaticity {
                x: values[0],
                y: values[1],
            }
        }
    }

    #[test]
    fn srgb_source_wire_goldens() {
        for (space, bits, expected) in [
            (ColorSpace::Rgb, 1, vec![0x01]),
            (ColorSpace::Gray, 15, vec![0x14, 0x37]),
        ] {
            let mut w = BitWriter::new(8);
            ColorEncoding::srgb(space).write_header(&mut w).unwrap();
            assert_eq!(w.bits_written(), bits);
            assert_eq!(Reader::new(w).bytes, expected);
        }
    }

    #[test]
    fn named_color_semantics_are_serialized_not_replaced() {
        for (tf, tf_id) in [
            (TransferFunction::Bt709, 1),
            (TransferFunction::Linear, 8),
            (TransferFunction::Srgb, 13),
            (TransferFunction::Pq, 16),
            (TransferFunction::Dci, 17),
            (TransferFunction::Hlg, 18),
        ] {
            for (wp, wp_id) in [
                (WhitePoint::D65, 1),
                (WhitePoint::E, 10),
                (WhitePoint::Dci, 11),
            ] {
                for (prim, prim_id) in [
                    (Primaries::Srgb, 1),
                    (Primaries::Bt2100, 9),
                    (Primaries::P3, 11),
                ] {
                    let f = ColorFields {
                        color_space: ColorSpace::Rgb,
                        white_point: wp,
                        primaries: Some(prim),
                        transfer_function: tf,
                        rendering_intent: RenderingIntent::Absolute,
                    };
                    let mut w = BitWriter::new(64);
                    ColorEncoding::Fields(f).write_header(&mut w).unwrap();
                    let count = w.bits_written();
                    let mut r = Reader::new(w);
                    assert_eq!(r.bits(2), 0); // nondefault, not ICC
                    assert_eq!(r.enumeration(), 0);
                    assert_eq!(r.enumeration(), wp_id);
                    assert_eq!(r.enumeration(), prim_id);
                    assert_eq!(r.bits(1), 0);
                    assert_eq!(r.enumeration(), tf_id);
                    assert_eq!(r.enumeration(), 3);
                    assert_eq!(r.pos, count);
                }
            }
        }
    }

    #[test]
    fn custom_negative_xy_and_gamma_rounding_boundaries() {
        for value in [
            -2097152, -1048576, -524288, -262144, -1, 0, 262143, 262144, 524288, 1048576, 2097151,
        ] {
            let xy = Chromaticity { x: value, y: value };
            xy.validate().unwrap();
            let mut w = BitWriter::new(8);
            xy.write(&mut w).unwrap();
            assert_eq!(Reader::new(w).xy(), xy);
        }
        assert_eq!(
            Chromaticity::from_xy(0.0000005, -0.0000005).unwrap(),
            Chromaticity { x: 1, y: -1 }
        );
        assert!(Chromaticity::from_xy(f64::NAN, 0.3).is_err());
        assert!(Chromaticity { x: 2097152, y: 0 }.validate().is_err());
        for g in [0, 1220, 10_000_001] {
            assert!(TransferFunction::Gamma(g).validate().is_err());
        }
        for g in [1221, 4_545_455, 10_000_000] {
            let mut f = ColorFields::srgb(ColorSpace::Gray);
            f.white_point = WhitePoint::Custom(Chromaticity {
                x: 345700,
                y: 358500,
            });
            f.transfer_function = TransferFunction::Gamma(g);
            let mut w = BitWriter::new(32);
            ColorEncoding::Fields(f.clone())
                .write_header(&mut w)
                .unwrap();
            let mut r = Reader::new(w);
            assert_eq!(r.bits(2), 0);
            assert_eq!(r.enumeration(), 1);
            assert_eq!(r.enumeration(), 2);
            assert_eq!(
                r.xy(),
                Chromaticity {
                    x: 345700,
                    y: 358500
                }
            );
            assert_eq!(r.bits(1), 1);
            assert_eq!(r.bits(24), g);
            assert_eq!(r.enumeration(), 1);
        }
        let f = ColorFields {
            color_space: ColorSpace::Rgb,
            white_point: WhitePoint::D65,
            primaries: Some(Primaries::Custom {
                red: Chromaticity {
                    x: 734700,
                    y: 265300,
                },
                green: Chromaticity { x: 0, y: 1000000 },
                blue: Chromaticity { x: 100, y: -77000 },
            }),
            transfer_function: TransferFunction::Linear,
            rendering_intent: RenderingIntent::Perceptual,
        };
        ColorEncoding::Fields(f)
            .write_header(&mut BitWriter::new(64))
            .unwrap();
    }

    #[test]
    fn invalid_semantics_leave_writer_untouched() {
        let mut f = ColorFields::srgb(ColorSpace::Gray);
        f.primaries = Some(Primaries::P3);
        let mut w = BitWriter::new(32);
        w.write(3, 5).unwrap();
        assert!(ColorEncoding::Fields(f).write_header(&mut w).is_err());
        assert_eq!(w.bits_written(), 3);
        assert!(ColorSpace::try_from(2).is_err());
        assert!(ColorSpace::try_from(3).is_err());
        assert!(ColorEncoding::icc(ColorSpace::Rgb, Vec::new()).is_err());
        for (signature, space, expected) in [
            (b"RGB ", ColorSpace::Rgb, 2),
            (b"GRAY", ColorSpace::Gray, 6),
        ] {
            let mut profile = vec![0; 132];
            profile[..4].copy_from_slice(&132u32.to_be_bytes());
            profile[16..20].copy_from_slice(signature);
            profile[20..24].copy_from_slice(b"XYZ ");
            profile[36..40].copy_from_slice(b"acsp");
            let encoding = ColorEncoding::icc(space, profile.clone()).unwrap();
            let mut w = BitWriter::new(1);
            encoding.write_header(&mut w).unwrap();
            assert_eq!(w.bits_written(), 4);
            assert_eq!(Reader::new(w).bytes, vec![expected]);
            let other = if space == ColorSpace::Rgb {
                ColorSpace::Gray
            } else {
                ColorSpace::Rgb
            };
            assert!(ColorEncoding::icc(other, profile).is_err());
        }
    }
}
