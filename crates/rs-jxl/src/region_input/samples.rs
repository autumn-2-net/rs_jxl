use crate::metadata::BitDepth;
use crate::modular::sample_convert::{SampleFormat, float_to_int_lossless};
use crate::{Error, PixelData, Result};

/// Borrowed native-endian typed samples, matching PixelData. F16 contains raw
/// binary16 words; F32 retains binary32 bits, including signed zero and NaNs.
#[derive(Clone, Copy, Debug)]
pub enum SampleSlice<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    F16(&'a [u16]),
    F32(&'a [f32]),
}

impl<'a> SampleSlice<'a> {
    pub fn len(self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) | Self::F16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub(super) fn from_pixels(pixels: &'a PixelData) -> Self {
        match pixels {
            PixelData::U8(v) => Self::U8(v),
            PixelData::U16(v) => Self::U16(v),
            PixelData::U32(v) => Self::U32(v),
            PixelData::F16(v) => Self::F16(v),
            PixelData::F32(v) => Self::F32(v),
        }
    }

    pub(super) fn tail(self, offset: usize) -> Result<Self> {
        let invalid = || Error::InvalidInput("region sample offset exceeds storage".into());
        Ok(match self {
            Self::U8(v) => Self::U8(v.get(offset..).ok_or_else(invalid)?),
            Self::U16(v) => Self::U16(v.get(offset..).ok_or_else(invalid)?),
            Self::U32(v) => Self::U32(v.get(offset..).ok_or_else(invalid)?),
            Self::F16(v) => Self::F16(v.get(offset..).ok_or_else(invalid)?),
            Self::F32(v) => Self::F32(v.get(offset..).ok_or_else(invalid)?),
        })
    }

    pub(super) fn validate_depth(self, depth: BitDepth) -> Result<()> {
        depth.validate()?;
        let valid = match self {
            Self::U8(_) => !depth.is_float() && depth.bits_per_sample <= 8,
            Self::U16(_) => !depth.is_float() && depth.bits_per_sample <= 16,
            Self::U32(_) => !depth.is_float(),
            Self::F16(_) => depth.bits_per_sample == 16 && depth.exponent_bits_per_sample == 5,
            Self::F32(_) => depth.is_float(),
        };
        if !valid {
            return Err(Error::InvalidInput(
                "region storage and declared sample depth disagree".into(),
            ));
        }
        Ok(())
    }

    // Called only after rectangular storage bounds and format validation.
    pub(super) fn encoded(self, index: usize, depth: BitDepth) -> Result<i32> {
        let value = match self {
            Self::U8(v) => u32::from(v[index]),
            Self::U16(v) => u32::from(v[index]),
            Self::U32(v) => v[index],
            Self::F16(v) => return Ok(i32::from(v[index])),
            Self::F32(v) => {
                if depth.bits_per_sample == 32 && depth.exponent_bits_per_sample == 8 {
                    return Ok(v[index].to_bits() as i32);
                }
                let mut result = [0];
                float_to_int_lossless(
                    &v[index..index + 1],
                    &mut result,
                    SampleFormat::Float {
                        bits: depth.bits_per_sample,
                        exponent_bits: depth.exponent_bits_per_sample,
                    },
                )?;
                return Ok(result[0]);
            }
        };
        if value > (1u32 << depth.bits_per_sample) - 1 {
            return Err(Error::InvalidInput(
                "region sample exceeds declared unsigned depth".into(),
            ));
        }
        Ok(value as i32)
    }
}
