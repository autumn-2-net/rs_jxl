// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Pixel-lossless frame routing and deferred reference policy. No scheduler dependency.
//! RoundtripPatchFrame re-enters EncodeFrame with Gradient and the original
//! auto-group/raw sampling inputs, not the main frame's resolved options.

use super::{
    encode::ModularOptions,
    predictor::Predictor,
    sample::{SamplingPolicy, SourcePolicy},
};
use crate::{Error, Image, Result};

/// CompressParams::buffering. Streaming requests still obey the source gates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i8)]
pub enum SourceBuffering {
    #[default]
    Auto = -1,
    FullFrame = 0,
    /// buffering=1: require at least one dimension greater than 2048.
    StreamingAbove2048 = 1,
    Streaming = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceFrameRoute {
    SharedFullFrame,
    /// Responsive lossless at tier >=1: whole-frame transforms, but each
    /// stream learns/encodes independently with NON-streaming entropy.
    IndependentFullFrame,
    StreamingFrameAreas,
}

/// enc_frame.cc::GetGroupSizeShift for pixel lossless, decoding tier zero.
pub fn automatic_group_size_shift(width: u32, height: u32) -> u8 {
    if width <= 128 && height <= 128 {
        0
    } else if width <= 256 && height <= 256 {
        1
    } else if width <= 400 && height <= 400 {
        2
    } else {
        1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceFrameGeometry {
    pub width: u32,
    pub height: u32,
    pub group_size_shift: u8,
    pub group_dim: usize,
    pub num_groups: usize,
    pub num_dc_groups: usize,
    pub num_streams: usize,
}

impl SourceFrameGeometry {
    /// enc_frame.cc::CanDoStreamingEncoding in the supported pixel-lossless
    /// domain: one pass, decoding tier zero, no noise/resampling/progressive
    /// overrides, no JPEG, color transform None. Also selects enc_frame.cc's
    /// full-frame ComputeTree gate for responsive lossless. Forced patches block streaming
    /// even if the later detector gate would disable them. Extra channels do
    /// not block nonresponsive lossless streaming. Geometry is the ROOT frame.
    pub fn pixel_route(
        &self,
        buffering: SourceBuffering,
        speed_tier: i8,
        forced_patches: bool,
        responsive: bool,
    ) -> SourceFrameRoute {
        // enc_frame.cc:1791: responsive lossless uses local trees unless below
        // Tortoise. This is not streaming; the entire global transform runs.
        if responsive {
            return if speed_tier < 1 {
                SourceFrameRoute::SharedFullFrame
            } else {
                SourceFrameRoute::IndependentFullFrame
            };
        }
        if buffering == SourceBuffering::FullFrame
            || (buffering == SourceBuffering::Auto && speed_tier < 1)
            || (buffering == SourceBuffering::StreamingAbove2048
                && self.width <= 2048
                && self.height <= 2048)
            || self.num_groups <= 8
            || forced_patches
        {
            SourceFrameRoute::SharedFullFrame
        } else {
            SourceFrameRoute::StreamingFrameAreas
        }
    }
}

/// ModularStreamId::Num for one pass, including 17 quant-table slots.
pub fn one_pass_geometry(
    width: u32,
    height: u32,
    requested_shift: Option<u8>,
) -> Result<SourceFrameGeometry> {
    if width == 0 || height == 0 || requested_shift.is_some_and(|shift| shift > 3) {
        return Err(Error::InvalidInput("invalid source frame geometry".into()));
    }
    let group_size_shift =
        requested_shift.unwrap_or_else(|| automatic_group_size_shift(width, height));
    let group_dim = 128usize << group_size_shift;
    let (w, h) = (width as usize, height as usize);
    let groups = w
        .div_ceil(group_dim)
        .checked_mul(h.div_ceil(group_dim))
        .ok_or(Error::Limit("source frame groups"))?;
    let dc_groups = w
        .div_ceil(group_dim * 8)
        .checked_mul(h.div_ceil(group_dim * 8))
        .ok_or(Error::Limit("source frame DC groups"))?;
    let num_streams = dc_groups
        .checked_mul(3)
        .and_then(|n| n.checked_add(groups))
        .and_then(|n| n.checked_add(18))
        .ok_or(Error::Limit("source frame streams"))?;
    Ok(SourceFrameGeometry {
        width,
        height,
        group_size_shift,
        group_dim,
        num_groups: groups,
        num_dc_groups: dc_groups,
        num_streams,
    })
}

#[derive(Clone, Debug)]
pub struct SourcePatchReferencePolicy {
    /// Projected with Gradient BEFORE selecting LZ77/WP. Sampling is deferred;
    /// this template must not be sent directly to the encoder.
    pub modular: ModularOptions,
    /// None retains the source auto sentinel until the atlas dimensions exist.
    pub group_size_shift: Option<u8>,
    /// Original CompressParams::buffering, not the root frame's resolved route.
    /// RoundtripPatchFrame disables patches but retains this input.
    pub buffering: SourceBuffering,
    /// Raw, unscaled learning_fraction. Geometry and metadata are filled from
    /// the atlas by resolve; never derive this from an existing SamplingPolicy.
    pub sampling: SourcePolicy,
}

#[derive(Clone, Debug)]
pub struct SourcePatchReference {
    pub modular: ModularOptions,
    pub group_size_shift: u8,
}

impl SourcePatchReferencePolicy {
    /// Resolve only encoder settings; retain signed integer samples, declared-
    /// format floating words and metadata unchanged. The caller keeps shared
    /// tree/global palette as the full-frame fallback, with responsive=false,
    /// one pass, no recursive
    /// patches, and reference slot 3. The atlas resolves its OWN streaming gate.
    pub fn resolve(&self, atlas: &Image) -> Result<SourcePatchReference> {
        if self.modular.predictor != Predictor::Gradient
            || self.sampling.predictor != Predictor::Gradient
            || self.sampling.responsive
            || self.sampling.decoding_speed_tier != 0
            || !self.modular.squeeze.is_empty()
        {
            return Err(Error::InvalidInput(
                "source patch reference requires nonresponsive Gradient policy at decoding tier zero".into(),
            ));
        }
        let metadata = atlas.metadata();
        metadata.validate()?;
        // Typed float atlas construction owns conversion/replay. Its private
        // planes contain declared-format words, not signed differences of their
        // bits. Extras are opaque None-blended words.
        // Image owns sample validation. Do not reinterpret empty PixelData or
        // apply automatic search's <=16-bit integer gate to reference metadata.
        // Frozen RoundtripPatchFrame forces ec_resampling=1, whereas its frame
        // header shifts this factor by metadata dim_shift. Native subsampled
        // atlas encoding is a separate adaptation, not yet source-equivalent.
        if metadata.extra_channels.iter().any(|ec| ec.dim_shift != 0) {
            return Err(Error::InvalidInput(
                "source patch reference subsampled-extra policy is not equivalent to frozen RoundtripPatchFrame".into(),
            ));
        }
        let geometry = one_pass_geometry(atlas.width(), atlas.height(), self.group_size_shift)?;
        let streaming =
            geometry.pixel_route(self.buffering, self.sampling.speed_tier, false, false)
                == SourceFrameRoute::StreamingFrameAreas;
        let mut source = self.sampling.clone();
        source.num_streams = geometry.num_streams;
        source.max_channel_size = geometry.group_dim;
        source.num_extra_channels = metadata.extra_channels.len();
        source.is_gray = atlas.color_channels() == 1;
        let sampling = SamplingPolicy::from_source(&source)?;
        let mut modular = self.modular.clone();
        modular.entropy_options.streaming_mode = streaming;
        if let Some(palette) = &mut modular.palette_policy {
            palette.groupwise = streaming;
        } else if streaming {
            return Err(Error::InvalidInput(
                "source streaming reference requires a palette orchestration policy".into(),
            ));
        }
        modular.previous_channels = sampling.previous_channels;
        modular.split_threshold = f64::from(sampling.node_threshold);
        modular.sampling = Some(sampling);
        modular.validate()?;
        Ok(SourcePatchReference {
            modular,
            group_size_shift: geometry.group_size_shift,
        })
    }
}
