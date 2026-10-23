// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Explicit lossless frame/layer contracts from frame_header.{h,cc}.
use crate::{Error, Image, Result, bit_writer::BitWriter, codestream::write_choice};

/// Header-only view; region providers need no dummy image allocation.
#[derive(Clone, Copy)]
pub(crate) struct Geometry<'a> {
    pub width: u32,
    pub height: u32,
    pub metadata: &'a crate::metadata::ImageMetadata,
}

impl<'a> Geometry<'a> {
    pub fn width(self) -> u32 {
        self.width
    }
    pub fn height(self) -> u32 {
        self.height
    }
    pub fn bits(self) -> u8 {
        self.metadata.bit_depth.bits_per_sample
    }
    pub fn metadata(self) -> &'a crate::metadata::ImageMetadata {
        self.metadata
    }
}

impl<'a> From<&'a Image> for Geometry<'a> {
    fn from(image: &'a Image) -> Self {
        Self {
            width: image.width(),
            height: image.height(),
            metadata: image.metadata(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum BlendMode {
    #[default]
    Replace = 0,
    Add = 1,
    Blend = 2,
    AlphaWeightedAdd = 3,
    Multiply = 4,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlendInfo {
    pub mode: BlendMode,
    pub source: u8,
    pub alpha_channel: u8,
    pub clamp: bool,
}
impl BlendInfo {
    pub(crate) fn validate(self, extras: usize) -> Result<()> {
        let uses_alpha = matches!(self.mode, BlendMode::Blend | BlendMode::AlphaWeightedAdd);
        if self.source > 3
            || (uses_alpha
                && (extras == 0
                    || usize::from(self.alpha_channel) >= extras
                    || self.alpha_channel > 10))
        {
            return Err(Error::InvalidInput(
                "invalid frame blending reference/alpha channel".into(),
            ));
        }
        if (!uses_alpha && self.alpha_channel != 0)
            || (!uses_alpha && self.mode != BlendMode::Multiply && self.clamp)
        {
            return Err(Error::InvalidInput(
                "unused blending fields must be default".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn write(self, w: &mut BitWriter, extras: usize, partial: bool) -> Result<()> {
        self.validate(extras)?;
        write_choice(w, self.mode as u32, [(0, 0), (1, 0), (2, 0), (3, 2)])?;
        let alpha = matches!(self.mode, BlendMode::Blend | BlendMode::AlphaWeightedAdd);
        if alpha {
            write_choice(
                w,
                u32::from(self.alpha_channel),
                [(0, 0), (1, 0), (2, 0), (3, 3)],
            )?;
        }
        if alpha || self.mode == BlendMode::Multiply {
            w.write(1, u64::from(self.clamp))?;
        }
        if self.mode != BlendMode::Replace || partial {
            w.write(2, u64::from(self.source))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameOptions {
    pub origin: (i32, i32),
    /// Zero is a true layer, not a substituted one-tick animation frame.
    pub duration: u32,
    pub timecode: u32,
    pub name: String,
    pub blend: BlendInfo,
    /// Empty applies the color blending rule to every extra channel.
    pub extra_blend: Vec<BlendInfo>,
    pub save_as_reference: u8,
    /// Reference-only frames are not displayed and must precede a visible frame.
    pub reference_only: bool,
    pub save_before_color_transform: bool,
}
impl FrameOptions {
    pub(crate) fn validate(&self, canvas: &Image, image: &Image, last: bool) -> Result<()> {
        self.validate_geometry(canvas.into(), image.into(), last)
    }

    pub(crate) fn validate_geometry(
        &self,
        canvas: Geometry<'_>,
        image: Geometry<'_>,
        last: bool,
    ) -> Result<()> {
        for origin in [self.origin.0, self.origin.1] {
            let packed = crate::modular::tree::pack_signed(origin);
            if !crate::codestream::FRAME_SIZE_CHOICES
                .into_iter()
                .any(|(offset, bits)| {
                    packed >= offset && u64::from(packed - offset) < (1u64 << bits)
                })
            {
                return Err(Error::InvalidInput(
                    "frame origin exceeds the signed header range".into(),
                ));
            }
        }
        let extras = canvas.metadata().extra_channels.len();
        if self.name.len() > 1071
            || self.save_as_reference > 3
            || (!self.extra_blend.is_empty() && self.extra_blend.len() != extras)
        {
            return Err(Error::InvalidInput(
                "invalid frame name/reference/extra blending count".into(),
            ));
        }
        if canvas.metadata().animation.is_none() && (self.duration != 0 || self.timecode != 0) {
            return Err(Error::InvalidInput(
                "duration/timecode requires animation metadata".into(),
            ));
        }
        if self.timecode != 0 && !canvas.metadata().animation.is_some_and(|a| a.timecodes) {
            return Err(Error::InvalidInput("timecodes are not enabled".into()));
        }
        if self.reference_only
            && (last || self.duration != 0 || self.timecode != 0 || self.origin != (0, 0))
        {
            return Err(Error::InvalidInput(
                "reference-only frame cannot be last, timed or offset".into(),
            ));
        }
        if self.reference_only
            && (self.blend != BlendInfo::default() || !self.extra_blend.is_empty())
        {
            return Err(Error::InvalidInput(
                "reference-only frame has no blending header".into(),
            ));
        }
        if last && self.save_as_reference != 0 {
            return Err(Error::InvalidInput(
                "last frame cannot save a reference".into(),
            ));
        }
        self.blend.validate(extras)?;
        for blend in &self.extra_blend {
            blend.validate(extras)?;
        }
        if self.reference_only
            && !self.save_before_color_transform
            && (image.width() < canvas.width() || image.height() < canvas.height())
        {
            return Err(Error::InvalidInput(
                "post-transform reference must cover canvas".into(),
            ));
        }
        let can_reference = !last && (self.duration == 0 || self.save_as_reference != 0);
        if self.save_before_color_transform
            && !self.reference_only
            && (!can_reference
                || self.partial_geometry(canvas, image)
                || self.blend.mode != BlendMode::Replace)
        {
            return Err(Error::InvalidInput(
                "pre-transform save needs an unblended full reference frame".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn partial(&self, canvas: &Image, image: &Image) -> bool {
        self.partial_geometry(canvas.into(), image.into())
    }

    pub(crate) fn partial_geometry(&self, canvas: Geometry<'_>, image: Geometry<'_>) -> bool {
        self.origin.0 > 0
            || self.origin.1 > 0
            || i64::from(self.origin.0) + i64::from(image.width()) < i64::from(canvas.width())
            || i64::from(self.origin.1) + i64::from(image.height()) < i64::from(canvas.height())
    }
    pub(crate) fn extra(&self, index: usize) -> BlendInfo {
        self.extra_blend.get(index).copied().unwrap_or(self.blend)
    }
}

/// Borrowed samples; no copying is required to describe a frame sequence.
pub struct Frame<'a> {
    pub image: &'a Image,
    pub options: FrameOptions,
}
impl<'a> Frame<'a> {
    pub fn new(image: &'a Image) -> Self {
        Self {
            image,
            options: FrameOptions::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout};

    #[test]
    fn origins_match_the_wire_range_before_any_encoding_work() {
        let image = Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![3])).unwrap();
        for origin in [-536_880_256, -1, 0, 1, 536_880_255] {
            for position in [(origin, 0), (0, origin)] {
                let frame = FrameOptions {
                    origin: position,
                    ..Default::default()
                };
                assert!(frame.validate(&image, &image, true).is_ok());
            }
        }
        for origin in [i32::MIN, -536_880_257, 536_880_256, i32::MAX] {
            for position in [(origin, 0), (0, origin)] {
                let frame = FrameOptions {
                    origin: position,
                    ..Default::default()
                };
                assert!(matches!(
                    frame.validate(&image, &image, true),
                    Err(Error::InvalidInput(_))
                ));
            }
        }
    }
}
