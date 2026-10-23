//! Typed samples and explicit image semantics. No implicit color conversion.
use crate::color::{ColorEncoding, ColorSpace};
use crate::metadata::{BitDepth, ExtraChannelInfo, ExtraChannelType, ImageMetadata};
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelLayout {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
}
impl PixelLayout {
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::GrayAlpha | Self::Rgba)
    }
}

/// Integer values are not rescaled. F16 contains IEEE binary16 bit patterns;
/// F32 preserves IEEE binary32 bits, including signed zero and NaN payloads.
#[derive(Debug, Clone)]
pub enum PixelData {
    U8(Vec<u8>),
    U16(Vec<u16>),
    /// The codestream integer depth is at most 31, not 32.
    U32(Vec<u32>),
    F16(Vec<u16>),
    F32(Vec<f32>),
}
impl PartialEq for PixelData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::U8(a), Self::U8(b)) => a == b,
            (Self::U16(a), Self::U16(b)) | (Self::F16(a), Self::F16(b)) => a == b,
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::F32(a), Self::F32(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
            }
            _ => false,
        }
    }
}
impl Eq for PixelData {}
impl PixelData {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) | Self::F16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub const fn bits(&self) -> u8 {
        match self {
            Self::U8(_) => 8,
            Self::U16(_) | Self::F16(_) => 16,
            Self::U32(_) | Self::F32(_) => 32,
        }
    }
    pub fn bit_depth(&self) -> BitDepth {
        let (bits_per_sample, exponent_bits_per_sample) = match self {
            Self::U8(_) => (8, 0),
            Self::U16(_) => (16, 0),
            Self::U32(_) => (31, 0),
            Self::F16(_) => (16, 5),
            Self::F32(_) => (32, 8),
        };
        BitDepth {
            bits_per_sample,
            exponent_bits_per_sample,
        }
    }
    pub(crate) fn get(&self, index: usize) -> i32 {
        match self {
            Self::U8(v) => i32::from(v[index]),
            Self::U16(v) | Self::F16(v) => i32::from(v[index]),
            Self::U32(v) => v[index] as i32,
            Self::F32(v) => v[index].to_bits() as i32,
        }
    }
    fn validate_channel(
        &self,
        depth: BitDepth,
        start: usize,
        stride: usize,
        count: usize,
    ) -> Result<()> {
        depth.validate()?;
        let storage = self.bit_depth();
        if let Self::F32(values) = self {
            if !depth.is_float() {
                return Err(Error::InvalidInput(
                    "floating sample buffer requires floating metadata".into(),
                ));
            }
            if depth != storage {
                let format = crate::modular::sample_convert::SampleFormat::Float {
                    bits: depth.bits_per_sample,
                    exponent_bits: depth.exponent_bits_per_sample,
                };
                for i in 0..count {
                    crate::modular::sample_convert::float_to_int_lossless(
                        &values[start + i * stride..start + i * stride + 1],
                        &mut [0],
                        format,
                    )?;
                }
            }
            return Ok(());
        }
        if storage.is_float() != depth.is_float()
            || (depth.is_float() && depth != storage)
            || (!depth.is_float() && depth.bits_per_sample > self.bits())
        {
            return Err(Error::InvalidInput(
                "sample storage and declared channel depth disagree".into(),
            ));
        }
        if !depth.is_float() {
            let max = (1u32 << depth.bits_per_sample) - 1;
            if (0..count).any(|i| self.get(start + i * stride) as u32 > max) {
                return Err(Error::InvalidInput(
                    "sample exceeds declared unsigned depth".into(),
                ));
            }
        }
        Ok(())
    }
    fn encoded(&self, index: usize, depth: BitDepth) -> i32 {
        if let Self::F32(values) = self
            && depth != self.bit_depth()
        {
            let mut result = [0];
            crate::modular::sample_convert::float_to_int_lossless(
                &values[index..index + 1],
                &mut result,
                crate::modular::sample_convert::SampleFormat::Float {
                    bits: depth.bits_per_sample,
                    exponent_bits: depth.exponent_bits_per_sample,
                },
            )
            .expect("validated immutable float samples");
            return result[0];
        }
        self.get(index)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    width: u32,
    height: u32,
    layout: PixelLayout,
    pixels: PixelData,
    metadata: ImageMetadata,
    extra_planes: Vec<PixelData>,
    // Encoder-only original-normalization planes (e.g. signed patch residuals).
    // Never exposed by public constructors or returned as a decoded image.
    modular_planes: Option<Vec<Vec<i32>>>,
}
impl Image {
    pub fn new(width: u32, height: u32, layout: PixelLayout, pixels: PixelData) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidInput(
                "image dimensions must be nonzero".into(),
            ));
        }
        let count = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(layout.channels()))
            .ok_or(Error::Limit("sample count overflow"))?;
        if pixels.len() != count {
            return Err(Error::InvalidInput(
                "sample count does not match dimensions/layout".into(),
            ));
        }
        let color = if layout.channels() < 3 {
            ColorSpace::Gray
        } else {
            ColorSpace::Rgb
        };
        let mut metadata = ImageMetadata::new(8, ColorEncoding::srgb(color))?;
        metadata.bit_depth = pixels.bit_depth();
        pixels.validate_channel(metadata.bit_depth, 0, 1, count)?;
        if layout.has_alpha() {
            let mut alpha = ExtraChannelInfo::new(ExtraChannelType::Alpha, 8)?;
            alpha.bit_depth = metadata.bit_depth;
            metadata.extra_channels.push(alpha);
        }
        Ok(Self {
            width,
            height,
            layout,
            pixels,
            metadata,
            extra_planes: Vec::new(),
            modular_planes: None,
        })
    }
    pub const fn width(&self) -> u32 {
        self.width
    }
    pub const fn height(&self) -> u32 {
        self.height
    }
    pub const fn layout(&self) -> PixelLayout {
        self.layout
    }
    pub fn pixels(&self) -> &PixelData {
        &self.pixels
    }
    pub fn bits(&self) -> u8 {
        self.metadata.bit_depth.bits_per_sample
    }
    pub fn metadata(&self) -> &ImageMetadata {
        &self.metadata
    }
    pub fn extra_planes(&self) -> &[PixelData] {
        &self.extra_planes
    }
    pub(crate) fn has_modular_planes(&self) -> bool {
        self.modular_planes.is_some()
    }
    pub fn channel_count(&self) -> usize {
        self.color_channels() + self.metadata.extra_channels.len()
    }
    pub fn color_channels(&self) -> usize {
        if self.layout.channels() < 3 { 1 } else { 3 }
    }
    pub fn channel_depth(&self, channel: usize) -> BitDepth {
        if channel < self.color_channels() {
            self.metadata.bit_depth
        } else {
            self.metadata.extra_channels[channel - self.color_channels()].bit_depth
        }
    }
    pub fn channel_shift(&self, channel: usize) -> u8 {
        if channel < self.color_channels() {
            0
        } else {
            self.metadata.extra_channels[channel - self.color_channels()].dim_shift
        }
    }
    pub fn channel_dimensions(&self, channel: usize) -> (usize, usize) {
        let scale = 1usize << self.channel_shift(channel);
        (
            (self.width as usize).div_ceil(scale),
            (self.height as usize).div_ceil(scale),
        )
    }
    pub(crate) fn modular_sample(&self, channel: usize, index: usize) -> i32 {
        if let Some(planes) = &self.modular_planes {
            return planes[channel][index];
        }
        let interleaved = self.layout.channels();
        if channel < interleaved {
            self.pixels
                .encoded(index * interleaved + channel, self.channel_depth(channel))
        } else {
            self.extra_planes[channel - interleaved].encoded(index, self.channel_depth(channel))
        }
    }
    pub(crate) fn with_modular_planes(
        &self,
        width: u32,
        height: u32,
        planes: Vec<Vec<i32>>,
    ) -> Result<Self> {
        if width == 0 || height == 0 || planes.len() != self.channel_count() {
            return Err(Error::Encoding(
                "signed frame planes do not match metadata".into(),
            ));
        }
        for (channel, plane) in planes.iter().enumerate() {
            let scale = 1usize << self.channel_shift(channel);
            let count = (width as usize)
                .div_ceil(scale)
                .checked_mul((height as usize).div_ceil(scale))
                .ok_or(Error::Limit("signed frame sample count"))?;
            if plane.len() != count {
                return Err(Error::Encoding(
                    "signed frame planes do not match metadata".into(),
                ));
            }
            let depth = self.channel_depth(channel);
            if depth.is_float() && depth.bits_per_sample < 32 {
                let mask = (1u32 << depth.bits_per_sample) - 1;
                if plane.iter().any(|&word| word as u32 & !mask != 0) {
                    return Err(Error::Encoding(
                        "floating frame word exceeds declared bit depth".into(),
                    ));
                }
            }
        }
        // Integer working planes may contain signed differences. Float planes
        // contain declared-format words, never differences of their bit patterns.
        // Extras retain their own depth and native subsampled dimensions.
        // Public samples remain absent from this private working representation.
        // All encoder sample access goes through modular_sample, not pixels().
        Ok(Self {
            width,
            height,
            layout: self.layout,
            pixels: PixelData::U8(Vec::new()),
            metadata: self.metadata.clone(),
            extra_planes: Vec::new(),
            modular_planes: Some(planes),
        })
    }
    /// Set metadata without changing values. Planar buffers follow any
    /// interleaved alpha; their dimensions are ceil(canvas / 2^dim_shift).
    pub fn with_metadata(
        mut self,
        metadata: ImageMetadata,
        extra_planes: Vec<PixelData>,
    ) -> Result<Self> {
        metadata.validate()?;
        let alpha = usize::from(self.layout.has_alpha());
        let color_space = if self.color_channels() == 1 {
            ColorSpace::Gray
        } else {
            ColorSpace::Rgb
        };
        if metadata.color_encoding.color_space() != color_space
            || metadata.extra_channels.len() != alpha + extra_planes.len()
        {
            return Err(Error::InvalidInput(
                "metadata does not match color/extra sample layout".into(),
            ));
        }
        let count = self.width as usize * self.height as usize;
        for c in 0..self.color_channels() {
            self.pixels
                .validate_channel(metadata.bit_depth, c, self.layout.channels(), count)?;
        }
        if alpha != 0 {
            let info = &metadata.extra_channels[0];
            if info.channel_type != ExtraChannelType::Alpha || info.dim_shift != 0 {
                return Err(Error::InvalidInput(
                    "interleaved alpha requires a full-resolution first alpha channel".into(),
                ));
            }
            self.pixels.validate_channel(
                info.bit_depth,
                self.layout.channels() - 1,
                self.layout.channels(),
                count,
            )?;
        }
        for (info, pixels) in metadata.extra_channels[alpha..].iter().zip(&extra_planes) {
            let scale = 1usize << info.dim_shift;
            let count = (self.width as usize)
                .div_ceil(scale)
                .checked_mul((self.height as usize).div_ceil(scale))
                .ok_or(Error::Limit("extra channel dimensions"))?;
            if pixels.len() != count {
                return Err(Error::InvalidInput(
                    "extra sample count does not match subsampled dimensions".into(),
                ));
            }
            pixels.validate_channel(info.bit_depth, 0, 1, count)?;
        }
        self.metadata = metadata;
        self.extra_planes = extra_planes;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn float_samples_compare_bits_not_numeric_equality() {
        let a = PixelData::F32(vec![f32::from_bits(0x7fc12345), -0.0]);
        assert_eq!(a, a.clone());
        assert_ne!(a, PixelData::F32(vec![f32::from_bits(0x7fc12345), 0.0]));
        assert_eq!(
            Image::new(2, 1, PixelLayout::Gray, a)
                .unwrap()
                .metadata()
                .bit_depth,
            BitDepth::floating(32, 8).unwrap()
        );
    }
    #[test]
    fn integer_boundaries_and_subsampled_extra_size() {
        assert!(Image::new(1, 1, PixelLayout::Gray, PixelData::U32(vec![1 << 31])).is_err());
        let image = Image::new(5, 3, PixelLayout::Gray, PixelData::U32(vec![0; 15])).unwrap();
        let mut meta = image.metadata().clone();
        let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
        extra.dim_shift = 1;
        meta.extra_channels.push(extra);
        let image = image
            .with_metadata(meta, vec![PixelData::U16(vec![0; 6])])
            .unwrap();
        assert_eq!(image.channel_dimensions(1), (3, 2));
    }
    #[test]
    fn float_working_planes_preserve_words_and_validate_declared_width() {
        let image = Image::new(
            2,
            1,
            PixelLayout::Gray,
            PixelData::F16(vec![0x8000, 0x7e01]),
        )
        .unwrap();
        let working = image
            .with_modular_planes(2, 1, vec![vec![0x8000, 0x7e01]])
            .unwrap();
        assert_eq!(working.modular_sample(0, 0), 0x8000);
        assert_eq!(working.modular_sample(0, 1), 0x7e01);
        assert!(image.with_modular_planes(1, 1, vec![vec![-1]]).is_err());
        assert!(
            image
                .with_modular_planes(1, 1, vec![vec![0x10000]])
                .is_err()
        );

        let image = Image::new(1, 1, PixelLayout::Gray, PixelData::F32(vec![-0.0])).unwrap();
        let working = image
            .with_modular_planes(1, 1, vec![vec![0x8000_0000u32 as i32]])
            .unwrap();
        assert_eq!(working.modular_sample(0, 0) as u32, 0x8000_0000);
    }
}
