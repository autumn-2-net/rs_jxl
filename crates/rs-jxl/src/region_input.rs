// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Borrowed pixel regions and bounded Modular frame-area encoding.
//!
//! Source contracts: encode.h::JxlChunkedFrameInputSource,
//! encode_internal.h::ScopedInputBuffer, and enc_frame.cc::EncodeFrameStreaming
//! at b866b3d. Area encoding never stages a full image; sequence writers may
//! explicitly opt into current-frame buffering for ineligible frames. Callbacks
//! are synchronous and caller-thread-only. Framing belongs to streaming.

#![forbid(unsafe_code)]

use crate::color::ColorSpace;
use crate::metadata::{BitDepth, ExtraChannelType, ImageMetadata};
use crate::{Error, Image, PixelData, PixelLayout, Result};

mod buffered;
mod load;
mod plan;
mod samples;

pub use plan::{FrameAreaPlan, SectionInfo, SectionKind};
pub use samples::SampleSlice;

pub(crate) use buffered::{buffered_sample_bytes, read_frame};

/// Image semantics without pixel ownership. Metadata is immutable after validation.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageDescriptor {
    width: u32,
    height: u32,
    layout: PixelLayout,
    metadata: ImageMetadata,
}

impl ImageDescriptor {
    pub fn new(
        width: u32,
        height: u32,
        layout: PixelLayout,
        metadata: ImageMetadata,
    ) -> Result<Self> {
        let descriptor = Self {
            width,
            height,
            layout,
            metadata,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Copies metadata only, never samples (including private working planes).
    pub fn from_image(image: &Image) -> Result<Self> {
        Self::new(
            image.width(),
            image.height(),
            image.layout(),
            image.metadata().clone(),
        )
    }

    pub fn validate(&self) -> Result<()> {
        self.metadata.validate()?;
        crate::codestream::validate_level10_geometry(self.geometry())?;
        let color = if self.color_channels() == 1 {
            ColorSpace::Gray
        } else {
            ColorSpace::Rgb
        };
        if self.metadata.color_encoding.color_space() != color {
            return Err(Error::InvalidInput(
                "descriptor color metadata and layout disagree".into(),
            ));
        }
        if self.layout.has_alpha()
            && !self.metadata.extra_channels.first().is_some_and(|extra| {
                extra.channel_type == ExtraChannelType::Alpha && extra.dim_shift == 0
            })
        {
            return Err(Error::InvalidInput(
                "interleaved alpha requires a full-resolution first alpha channel".into(),
            ));
        }
        Ok(())
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
    pub fn metadata(&self) -> &ImageMetadata {
        &self.metadata
    }
    pub fn bits(&self) -> u8 {
        self.metadata.bit_depth.bits_per_sample
    }
    pub fn color_channels(&self) -> usize {
        if self.layout.channels() < 3 { 1 } else { 3 }
    }
    pub fn channel_count(&self) -> usize {
        self.color_channels() + self.metadata.extra_channels.len()
    }

    pub fn channel_depth(&self, channel: usize) -> Result<BitDepth> {
        if channel < self.color_channels() {
            Ok(self.metadata.bit_depth)
        } else {
            self.metadata
                .extra_channels
                .get(channel - self.color_channels())
                .map(|extra| extra.bit_depth)
                .ok_or_else(|| Error::InvalidInput("descriptor channel index out of range".into()))
        }
    }

    pub fn channel_shift(&self, channel: usize) -> Result<u8> {
        if channel < self.color_channels() {
            Ok(0)
        } else {
            self.metadata
                .extra_channels
                .get(channel - self.color_channels())
                .map(|extra| extra.dim_shift)
                .ok_or_else(|| Error::InvalidInput("descriptor channel index out of range".into()))
        }
    }

    pub fn channel_dimensions(&self, channel: usize) -> Result<(usize, usize)> {
        let scale = 1usize << self.channel_shift(channel)?;
        Ok((
            (self.width as usize).div_ceil(scale),
            (self.height as usize).div_ceil(scale),
        ))
    }

    pub(crate) fn geometry(&self) -> crate::frame::Geometry<'_> {
        crate::frame::Geometry {
            width: self.width,
            height: self.height,
            metadata: &self.metadata,
        }
    }

    fn plane_layout(&self, plane: RegionPlane) -> Result<(usize, usize, usize, usize)> {
        match plane {
            RegionPlane::Color => Ok((
                self.width as usize,
                self.height as usize,
                0,
                self.layout.channels(),
            )),
            RegionPlane::Extra(extra) => {
                if extra == 0 && self.layout.has_alpha() {
                    return Err(Error::InvalidInput(
                        "interleaved alpha is requested with color".into(),
                    ));
                }
                let channel = self
                    .color_channels()
                    .checked_add(extra)
                    .ok_or(Error::Limit("region channel index"))?;
                let (width, height) = self.channel_dimensions(channel)?;
                Ok((width, height, channel, 1))
            }
        }
    }

    pub fn validate_request(&self, request: RegionRequest) -> Result<()> {
        let (width, height, _, _) = self.plane_layout(request.plane)?;
        request.rect.validate_within(width, height)
    }
}

/// Rectangles are in the requested plane's native sample coordinates. No
/// upsampling, orientation rewrite, alpha cleanup or color conversion occurs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegionRect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl RegionRect {
    fn validate_within(self, width: usize, height: usize) -> Result<()> {
        if self.width == 0
            || self.height == 0
            || self.x.checked_add(self.width).is_none_or(|end| end > width)
            || self
                .y
                .checked_add(self.height)
                .is_none_or(|end| end > height)
        {
            return Err(Error::InvalidInput("region lies outside its plane".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegionPlane {
    /// Interleaved Gray/GrayAlpha/Rgb/Rgba according to the descriptor.
    Color,
    /// Metadata extra-channel index, excluding alpha already present in Color.
    Extra(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegionRequest {
    pub plane: RegionPlane,
    pub rect: RegionRect,
}

#[derive(Clone, Copy, Debug)]
pub struct RegionView<'a> {
    /// Starts at the requested top-left sample; trailing storage is permitted.
    pub samples: SampleSlice<'a>,
    /// Native sample elements between rows, NOT bytes or pixels. Color rows
    /// contain rect.width * layout.channels() interleaved elements.
    pub row_stride: usize,
}

/// A provider may borrow its reusable storage until this lease is dropped.
/// The release callback executes once on success, errors, or stack unwinding.
pub struct RegionLease<'a> {
    view: RegionView<'a>,
    release: Option<Box<dyn FnOnce() + 'a>>,
}

impl<'a> RegionLease<'a> {
    pub fn borrowed(view: RegionView<'a>) -> Self {
        Self {
            view,
            release: None,
        }
    }

    pub fn with_release(view: RegionView<'a>, release: impl FnOnce() + 'a) -> Self {
        Self {
            view,
            release: Some(Box::new(release)),
        }
    }

    /// The view cannot outlive a borrow of the lease, even if its backing
    /// storage lives longer. Consumers must not retain samples after release.
    pub fn view(&self) -> RegionView<'_> {
        self.view
    }
}

impl Drop for RegionLease<'_> {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}

/// Requests and releases run on the caller thread, with at most one live lease.
/// No Send/Sync requirement is imposed on providers. Samples must remain stable
/// for the frame. The Modular loader requests tiles no larger than 2048 per
/// native plane axis; no promise of one callback per DC area is made.
pub trait RegionSource {
    fn descriptor(&self) -> &ImageDescriptor;
    fn acquire(&mut self, request: RegionRequest) -> Result<RegionLease<'_>>;
}

/// Zero-copy input adapter for ordinary typed Images. Signed encoder-private
/// patch planes are intentionally not a public native sample representation.
pub struct ImageRegionSource<'a> {
    image: &'a Image,
    descriptor: ImageDescriptor,
}

impl<'a> ImageRegionSource<'a> {
    pub fn new(image: &'a Image) -> Result<Self> {
        if image.has_modular_planes() {
            return Err(Error::InvalidInput(
                "private Modular working images are not region input".into(),
            ));
        }
        Ok(Self {
            image,
            descriptor: ImageDescriptor::from_image(image)?,
        })
    }

    pub fn descriptor(&self) -> &ImageDescriptor {
        &self.descriptor
    }
}

impl RegionSource for ImageRegionSource<'_> {
    fn descriptor(&self) -> &ImageDescriptor {
        &self.descriptor
    }

    fn acquire(&mut self, request: RegionRequest) -> Result<RegionLease<'_>> {
        self.descriptor.validate_request(request)?;
        let (width, _, _, components) = self.descriptor.plane_layout(request.plane)?;
        let pixels: &PixelData = match request.plane {
            RegionPlane::Color => self.image.pixels(),
            RegionPlane::Extra(extra) => {
                &self.image.extra_planes()[extra - usize::from(self.image.layout().has_alpha())]
            }
        };
        let stride = width
            .checked_mul(components)
            .ok_or(Error::Limit("region row stride"))?;
        let offset = request
            .rect
            .y
            .checked_mul(stride)
            .and_then(|n| {
                request
                    .rect
                    .x
                    .checked_mul(components)
                    .and_then(|x| n.checked_add(x))
            })
            .ok_or(Error::Limit("region sample offset"))?;
        Ok(RegionLease::borrowed(RegionView {
            samples: SampleSlice::from_pixels(pixels).tail(offset)?,
            row_stride: stride,
        }))
    }
}

#[cfg(test)]
mod tests;
