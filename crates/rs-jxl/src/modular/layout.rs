// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Full-frame integer Modular progressive layout, libjxl b866b3d source map:
//! - squeeze.cc::DefaultSqueezeParameters (preview size 8), MetaSqueeze;
//!   enc_squeeze.cc::FwdSqueeze residual insertion, using transforms.rs kernels.
//! - enc_modular.cc::ComputeEncodingData default-script shift cap (including
//!   resize(i - 1)), small-channel global prefix, DC/AC stream enumeration;
//!   PrepareStreamParams shift brackets and nominal-rectangle clipping.
//! - frame_header.h::Passes::GetDownsamplingBracket; frame_header.cc validation;
//!   dec_modular.h::ModularStreamId; frame_dimensions.h group sizes.
//! - enc_frame.cc::EncodeFrameStreaming DC-sized frame areas;
//!   enc_modular.cc::ComputeEncodingData area-relative to absolute stream IDs.
//!
//! Parent owns frame/pass/squeeze wire fields, responsive/headroom gates, RCT,
//! tree/entropy, and transport order. Stream IDs here are MA property IDs, NOT
//! TOC indices. Empty DC/AC streams remain in the plan. No local squeeze is
//! substituted for the source's global transform. No lossy quantization,
//! arbitrary unaligned frame-area crops, meta-channel squeezing, or VarDCT splitting.
//! The planner checks representable stream IDs and fallible allocation, without
//! an extra stream-count cap. Forward scalar arithmetic limits are transforms.rs's.

#![forbid(unsafe_code)]

use super::group::{self, Channel};
use super::transforms;
use crate::{Error, Result};

const QUANT_TABLES: usize = 17; // quant_weights.h::QuantTable::DCT128X256 + 1

fn validate_stream_ids(groups: usize, dc_groups: usize, passes: usize) -> Result<()> {
    // MA property 1 stores the stream ID as a signed 32-bit value.
    dc_groups
        .checked_mul(3)
        .and_then(|n| n.checked_add(QUANT_TABLES))
        .and_then(|n| groups.checked_mul(passes).and_then(|g| n.checked_add(g)))
        .filter(|&last| last <= i32::MAX as usize)
        .ok_or(Error::Limit("Modular stream IDs exceed i32"))?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelShape {
    pub width: usize,
    pub height: usize,
    pub hshift: i32,
    pub vshift: i32,
}

impl From<&Channel> for ChannelShape {
    fn from(c: &Channel) -> Self {
        Self {
            width: c.width,
            height: c.height,
            hshift: c.hshift,
            vshift: c.vshift,
        }
    }
}

fn validate_shapes(channels: &[ChannelShape], meta: usize) -> Result<()> {
    if meta > channels.len() || channels.len() > i32::MAX as usize {
        return Err(Error::InvalidInput(
            "invalid Modular channel/meta count".into(),
        ));
    }
    for (i, c) in channels.iter().enumerate() {
        c.width
            .checked_mul(c.height)
            .ok_or(Error::Limit("channel sample count"))?;
        if (i < meta && (c.hshift >= 0 || c.vshift >= 0))
            || (i >= meta && (!(0..=31).contains(&c.hshift) || !(0..=31).contains(&c.vshift)))
        {
            return Err(Error::InvalidInput(
                "invalid Modular channel shifts/meta prefix".into(),
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SqueezeParams {
    pub horizontal: bool,
    pub in_place: bool,
    pub begin: usize,
    pub channels: usize,
}

/// Literal default script. The chroma prelude checks only channel 1's geometry,
/// exactly as the source does; no RGB/color-content inference is made here.
pub fn default_squeeze_parameters(
    channels: &[ChannelShape],
    meta: usize,
) -> Result<Vec<SqueezeParams>> {
    validate_shapes(channels, meta)?;
    let Some(first) = channels.get(meta) else {
        return Ok(Vec::new());
    };
    let (mut w, mut h) = (first.width, first.height);
    if w == 0 || h == 0 {
        return Err(Error::InvalidInput("empty first squeeze channel".into()));
    }
    let count = channels.len() - meta;
    let wide = w > h;
    let mut parameters = Vec::new();
    if count > 2 && (channels[meta + 1].width, channels[meta + 1].height) == (w, h) {
        parameters.push(SqueezeParams {
            horizontal: true,
            in_place: false,
            begin: meta + 1,
            channels: 2,
        });
        parameters.push(SqueezeParams {
            horizontal: false,
            in_place: false,
            begin: meta + 1,
            channels: 2,
        });
    }
    let mut p = SqueezeParams {
        horizontal: false,
        in_place: true,
        begin: meta,
        channels: count,
    };
    if !wide && h > 8 {
        parameters.push(p);
        h = h.div_ceil(2);
    }
    while w > 8 || h > 8 {
        if w > 8 {
            p.horizontal = true;
            parameters.push(p);
            w = w.div_ceil(2);
        }
        if h > 8 {
            p.horizontal = false;
            parameters.push(p);
            h = h.div_ceil(2);
        }
    }
    Ok(parameters)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SqueezePlan {
    pub parameters: Vec<SqueezeParams>,
    /// True: source emits zero squeeze parameters, and decoder derives default.
    /// False: the shift cap truncated the script; emit these explicit parameters.
    /// An empty parameters vector means no transform, regardless of this flag.
    pub use_default: bool,
}

/// ComputeEncodingData's cap prevents an anisotropic group tile shrinking to
/// zero. Caller applies source responsive/groupwise/level gates before calling.
pub fn plan_squeeze(
    channels: &[ChannelShape],
    meta: usize,
    group_size_shift: u8,
) -> Result<SqueezePlan> {
    if group_size_shift > 3 {
        return Err(Error::InvalidInput("group size shift must be 0..3".into()));
    }
    let mut plan = SqueezePlan {
        parameters: default_squeeze_parameters(channels, meta)?,
        use_default: true,
    };
    let cap = 7 + usize::from(group_size_shift);
    let (mut hshift, mut vshift) = (0usize, 0usize);
    for i in 0..plan.parameters.len() {
        let p = plan.parameters[i];
        if p.horizontal {
            hshift += 1;
        } else {
            vshift += 1;
        }
        let dc_boost = if hshift.min(vshift) >= 3 { 3 } else { 0 };
        if hshift.max(vshift) > cap + dc_boost {
            // Source intentionally removes the preceding pass as well.
            plan.parameters
                .truncate(i.checked_sub(1).ok_or(Error::Limit("squeeze shift cap"))?);
            plan.use_default = false;
            break;
        }
    }
    if !plan.use_default && plan.parameters.iter().any(|p| p.channels > 19) {
        return Err(Error::Limit(
            "explicit squeeze channel count exceeds wire range",
        ));
    }
    squeeze_shapes(channels, meta, &plan.parameters)?;
    Ok(plan)
}

fn squeeze_range(channels: &[ChannelShape], meta: usize, p: SqueezeParams) -> Result<usize> {
    let end = p
        .begin
        .checked_add(p.channels)
        .filter(|&end| p.channels != 0 && p.begin >= meta && end <= channels.len())
        .ok_or_else(|| Error::InvalidInput("squeeze requires a non-meta channel range".into()))?;
    if channels
        .len()
        .checked_add(p.channels)
        .filter(|&n| n <= i32::MAX as usize)
        .is_none()
    {
        return Err(Error::Limit("squeeze channel count"));
    }
    for c in &channels[p.begin..end] {
        if c.width == 0 || c.height == 0 || c.hshift > 30 || c.vshift > 30 {
            return Err(Error::InvalidInput(
                "squeezing an empty channel or a shift above 30".into(),
            ));
        }
    }
    Ok(end)
}

/// Metadata-only execution with source residual order. Explicit empty list is
/// a no-op here, not an instruction to regenerate the default script.
pub fn squeeze_shapes(
    channels: &[ChannelShape],
    meta: usize,
    parameters: &[SqueezeParams],
) -> Result<Vec<ChannelShape>> {
    validate_shapes(channels, meta)?;
    if parameters.len() > 296 {
        return Err(Error::Limit("squeeze parameter count"));
    }
    let mut result = channels.to_vec();
    for &p in parameters {
        let end = squeeze_range(&result, meta, p)?;
        let offset = if p.in_place { end } else { result.len() };
        let mut residuals = Vec::with_capacity(p.channels);
        for c in &mut result[p.begin..end] {
            let mut residual = *c;
            if p.horizontal {
                c.width = c.width.div_ceil(2);
                c.hshift += 1;
                residual.width -= c.width;
                residual.hshift = c.hshift;
            } else {
                c.height = c.height.div_ceil(2);
                c.vshift += 1;
                residual.height -= c.height;
                residual.vshift = c.vshift;
            }
            residuals.push(residual);
        }
        result.splice(offset..offset, residuals);
    }
    Ok(result)
}

/// Forward sample execution before group partitioning. Source append/in-place
/// semantics include zero-sized residual channels (e.g. a one-pixel axis).
/// All errors leave input untouched. No wire metadata is emitted here.
pub fn apply_squeeze(
    channels: &mut Vec<Channel>,
    meta: usize,
    parameters: &[SqueezeParams],
) -> Result<()> {
    group::validate_layout(channels, meta)?;
    let shapes: Vec<_> = channels.iter().map(ChannelShape::from).collect();
    let expected = squeeze_shapes(&shapes, meta, parameters)?;
    let mut candidate = channels.clone();
    for &p in parameters {
        let end = p.begin + p.channels;
        let offset = if p.in_place { end } else { candidate.len() };
        let mut residuals = Vec::with_capacity(p.channels);
        for c in &mut candidate[p.begin..end] {
            let split = transforms::squeeze_forward(&c.samples, c.width, c.height, p.horizontal)?;
            if p.horizontal {
                c.hshift += 1;
            } else {
                c.vshift += 1;
            }
            c.width = split.average_width;
            c.height = split.average_height;
            c.samples = split.average;
            residuals.push(Channel {
                width: split.residual_width,
                height: split.residual_height,
                hshift: c.hshift,
                vshift: c.vshift,
                samples: split.residual,
            });
        }
        candidate.splice(offset..offset, residuals);
    }
    debug_assert_eq!(
        candidate.iter().map(ChannelShape::from).collect::<Vec<_>>(),
        expected
    );
    *channels = candidate;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Downsample {
    pub factor: u8,
    pub last_pass: u8,
}

/// Only the spatial fields used by Modular. Parent serializes frame Passes;
/// coefficient shift fields belong to VarDCT, not this layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Passes {
    pub num_passes: u8,
    pub downsample: Vec<Downsample>,
}

impl Default for Passes {
    fn default() -> Self {
        Self {
            num_passes: 1,
            downsample: Vec::new(),
        }
    }
}

impl Passes {
    /// Spatial brackets of enc_frame.cc's dc_vlf_lf_full_ac progressive mode.
    /// Responsive squeeze alone does not automatically request these 3 passes.
    pub fn progressive() -> Self {
        Self {
            num_passes: 3,
            downsample: vec![
                Downsample {
                    factor: 4,
                    last_pass: 0,
                },
                Downsample {
                    factor: 2,
                    last_pass: 1,
                },
            ],
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !(1..=11).contains(&self.num_passes)
            || self.downsample.len() > 4
            || self.downsample.len() > usize::from(self.num_passes)
            || (self.num_passes == 1 && !self.downsample.is_empty())
        {
            return Err(Error::InvalidInput(
                "invalid Modular spatial pass count".into(),
            ));
        }
        for (i, d) in self.downsample.iter().enumerate() {
            if ![1, 2, 4, 8].contains(&d.factor)
                || d.last_pass >= self.num_passes
                || d.last_pass > 7
                || (i > 0
                    && (d.factor >= self.downsample[i - 1].factor
                        || d.last_pass <= self.downsample[i - 1].last_pass))
            {
                return Err(Error::InvalidInput(
                    "invalid decreasing downsample/increasing pass sequence".into(),
                ));
            }
        }
        Ok(())
    }

    /// Inclusive bounds; min > max is a valid empty spatial pass.
    pub fn bracket(&self, pass: usize) -> Result<ShiftRange> {
        self.validate()?;
        if pass >= usize::from(self.num_passes) {
            return Err(Error::InvalidInput("pass index out of range".into()));
        }
        let mut range = ShiftRange { min: 3, max: 2 };
        for i in 0..=pass {
            for d in &self.downsample {
                if i == usize::from(d.last_pass) {
                    range.min = d.factor.trailing_zeros() as i32;
                }
            }
            if i == usize::from(self.num_passes) - 1 {
                range.min = 0;
            }
            if i == pass {
                return Ok(range);
            }
            range.max = range.min - 1;
        }
        unreachable!()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShiftRange {
    pub min: i32,
    pub max: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamKind {
    Dc { group: usize },
    Ac { group: usize, pass: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelTile {
    /// Index in the globally transformed channel vector, before local transforms.
    pub channel: usize,
    /// In channel coordinates, NOT original full-resolution coordinates.
    pub rect: Rect,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamPlan {
    pub kind: StreamKind,
    pub stream_id: usize,
    /// Nominal full-resolution group rectangle. Edge sizes are not pre-clipped.
    pub rect: Rect,
    pub shifts: ShiftRange,
    pub channels: Vec<ChannelTile>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartitionPlan {
    /// Keep channels[..global_channels] in DCGlobal, including small non-meta
    /// channels until the FIRST large channel. Do not filter small later ones.
    pub global_channels: usize,
    pub num_groups: usize,
    pub num_dc_groups: usize,
    /// Source preparation order: all DC, then AC group-major/pass-minor.
    /// Parent may reorder by TOC; stream_id is not a TOC entry index.
    pub streams: Vec<StreamPlan>,
}

fn stream_tiles(
    channels: &[ChannelShape],
    begin: usize,
    rect: Rect,
    shifts: ShiftRange,
) -> Vec<ChannelTile> {
    let mut result = Vec::new();
    for (channel, c) in channels.iter().enumerate().skip(begin) {
        let shift = c.hshift.min(c.vshift);
        if shift < shifts.min || shift > shifts.max {
            continue;
        }
        let x = rect.x >> c.hshift;
        let y = rect.y >> c.vshift;
        let width = (rect.width >> c.hshift).min(c.width.saturating_sub(x));
        let height = (rect.height >> c.vshift).min(c.height.saturating_sub(y));
        if width != 0 && height != 0 {
            result.push(ChannelTile {
                channel,
                rect: Rect {
                    x,
                    y,
                    width,
                    height,
                },
            });
        }
    }
    result
}

/// Whole-frame non-streaming partition. Dimensions are the pre-squeeze Modular
/// frame dimensions, with no upsampling/crop offsets. The original shape must
/// fit its declared shifts. Rejects layouts with uncovered samples (normally
/// prevented by the source squeeze cap), rather than silently discarding them.
pub fn partition_channels(
    channels: &[ChannelShape],
    meta: usize,
    width: usize,
    height: usize,
    group_size_shift: u8,
    passes: &Passes,
) -> Result<PartitionPlan> {
    partition(
        channels,
        meta,
        width,
        height,
        group_size_shift,
        passes,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn partition(
    channels: &[ChannelShape],
    meta: usize,
    width: usize,
    height: usize,
    group_size_shift: u8,
    passes: &Passes,
    keep_small_global: bool,
) -> Result<PartitionPlan> {
    validate_shapes(channels, meta)?;
    passes.validate()?;
    if width == 0 || height == 0 || group_size_shift > 3 {
        return Err(Error::InvalidInput(
            "invalid Modular frame/group dimensions".into(),
        ));
    }
    for c in &channels[meta..] {
        if c.width > width.div_ceil(1usize << c.hshift)
            || c.height > height.div_ceil(1usize << c.vshift)
        {
            return Err(Error::InvalidInput(
                "channel dimensions exceed shifted frame".into(),
            ));
        }
    }
    let group_dim = 128usize << group_size_shift;
    let dc_dim = group_dim * 8;
    let gx = width.div_ceil(group_dim);
    let gy = height.div_ceil(group_dim);
    let dx = width.div_ceil(dc_dim);
    let dy = height.div_ceil(dc_dim);
    let num_groups = gx.checked_mul(gy).ok_or(Error::Limit("AC group count"))?;
    let num_dc_groups = dx.checked_mul(dy).ok_or(Error::Limit("DC group count"))?;
    let total = num_groups
        .checked_mul(usize::from(passes.num_passes))
        .and_then(|n| n.checked_add(num_dc_groups))
        .ok_or(Error::Limit("Modular layout stream count"))?;
    validate_stream_ids(num_groups, num_dc_groups, usize::from(passes.num_passes))?;
    let global_channels = if keep_small_global {
        channels
            .iter()
            .enumerate()
            .skip(meta)
            .find(|(_, c)| c.width > group_dim || c.height > group_dim)
            .map_or(channels.len(), |(i, _)| i)
    } else {
        meta
    };
    let mut streams = Vec::new();
    streams
        .try_reserve_exact(total)
        .map_err(|_| Error::Limit("Modular stream layout allocation"))?;
    for group in 0..num_dc_groups {
        let rect = Rect {
            x: (group % dx) * dc_dim,
            y: (group / dx) * dc_dim,
            width: dc_dim,
            height: dc_dim,
        };
        let shifts = ShiftRange { min: 3, max: 1000 };
        streams.push(StreamPlan {
            kind: StreamKind::Dc { group },
            stream_id: 1 + num_dc_groups + group,
            rect,
            shifts,
            channels: stream_tiles(channels, global_channels, rect, shifts),
        });
    }
    for group in 0..num_groups {
        let rect = Rect {
            x: (group % gx) * group_dim,
            y: (group / gx) * group_dim,
            width: group_dim,
            height: group_dim,
        };
        for pass in 0..usize::from(passes.num_passes) {
            let shifts = passes.bracket(pass)?;
            streams.push(StreamPlan {
                kind: StreamKind::Ac { group, pass },
                stream_id: 1 + 3 * num_dc_groups + QUANT_TABLES + num_groups * pass + group,
                rect,
                shifts,
                channels: stream_tiles(channels, global_channels, rect, shifts),
            });
        }
    }
    let mut covered = vec![0usize; channels.len()];
    for stream in &streams {
        for tile in &stream.channels {
            covered[tile.channel] = covered[tile.channel]
                .checked_add(tile.rect.width * tile.rect.height)
                .ok_or(Error::Limit("partition sample coverage"))?;
        }
    }
    for (i, c) in channels.iter().enumerate().skip(global_channels) {
        if covered[i] != c.width * c.height {
            return Err(Error::InvalidInput(
                "shifted partition leaves samples uncovered; check squeeze cap".into(),
            ));
        }
    }
    Ok(PartitionPlan {
        global_channels,
        num_groups,
        num_dc_groups,
        streams,
    })
}

/// Raster-ordered DC-sized frame areas from enc_frame.cc::EncodeFrameStreaming.
/// Rectangles are in frame pixels and clipped only at the right/bottom edge.
/// This is a layout helper, not the streaming eligibility decision.
pub fn frame_areas(width: usize, height: usize, group_size_shift: u8) -> Result<Vec<Rect>> {
    if width == 0 || height == 0 || group_size_shift > 3 {
        return Err(Error::InvalidInput(
            "invalid streaming frame dimensions".into(),
        ));
    }
    let group_dim = 128usize << group_size_shift;
    let dc_dim = 1024usize << group_size_shift;
    let columns = width.div_ceil(dc_dim);
    let count = columns
        .checked_mul(height.div_ceil(dc_dim))
        .ok_or(Error::Limit("DC group count"))?;
    width
        .div_ceil(group_dim)
        .checked_mul(height.div_ceil(group_dim))
        .and_then(|n| n.checked_add(count))
        .ok_or(Error::Limit("Modular layout stream count"))?;
    let mut areas = Vec::new();
    areas
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("frame area allocation"))?;
    for group in 0..count {
        let x = group % columns * dc_dim;
        let y = group / columns * dc_dim;
        areas.push(Rect {
            x,
            y,
            width: dc_dim.min(width - x),
            height: dc_dim.min(height - y),
        });
    }
    Ok(areas)
}

/// PrepareStreamParams(groupwise=true): do not retain any small non-meta
/// channel prefix. Samples/rectangles are area-relative; stream kinds and MA
/// IDs are frame-absolute. Only a complete, aligned DC frame area is accepted.
///
/// Deliberate source bug fix: frozen enc_modular.cc divides DC offsets by 2048,
/// which only works for group_size_shift=1. Use the frame's actual dc_group_dim
/// so shifted extras address the same DC group as the TOC and decoder.
#[allow(clippy::too_many_arguments)]
pub fn partition_frame_area(
    channels: &[ChannelShape],
    meta: usize,
    width: usize,
    height: usize,
    group_size_shift: u8,
    passes: &Passes,
    area: Rect,
) -> Result<PartitionPlan> {
    if group_size_shift > 3 || width == 0 || height == 0 {
        return Err(Error::InvalidInput(
            "invalid streaming frame dimensions".into(),
        ));
    }
    let group_dim = 128usize << group_size_shift;
    let dc_dim = group_dim * 8;
    if area.x >= width
        || area.y >= height
        || !area.x.is_multiple_of(dc_dim)
        || !area.y.is_multiple_of(dc_dim)
        || area.width != dc_dim.min(width - area.x)
        || area.height != dc_dim.min(height - area.y)
    {
        return Err(Error::InvalidInput(
            "streaming area must be one aligned clipped DC group".into(),
        ));
    }
    let mut plan = partition(
        channels,
        meta,
        area.width,
        area.height,
        group_size_shift,
        passes,
        false,
    )?;
    let local_columns = area.width.div_ceil(group_dim);
    let full_columns = width.div_ceil(group_dim);
    let dc_columns = width.div_ceil(dc_dim);
    let num_groups = full_columns
        .checked_mul(height.div_ceil(group_dim))
        .ok_or(Error::Limit("AC group count"))?;
    let num_dc_groups = dc_columns
        .checked_mul(height.div_ceil(dc_dim))
        .ok_or(Error::Limit("DC group count"))?;
    num_groups
        .checked_mul(usize::from(passes.num_passes))
        .and_then(|n| n.checked_add(num_dc_groups))
        .ok_or(Error::Limit("Modular layout stream count"))?;
    validate_stream_ids(num_groups, num_dc_groups, usize::from(passes.num_passes))?;
    let dc_group = area.y / dc_dim * dc_columns + area.x / dc_dim;
    for stream in &mut plan.streams {
        match stream.kind {
            StreamKind::Dc { .. } => {
                stream.kind = StreamKind::Dc { group: dc_group };
                stream.stream_id = 1 + num_dc_groups + dc_group;
            }
            StreamKind::Ac { group, pass } => {
                let x = area.x / group_dim + group % local_columns;
                let y = area.y / group_dim + group / local_columns;
                let group = y * full_columns + x;
                stream.kind = StreamKind::Ac { group, pass };
                stream.stream_id = 1 + 3 * num_dc_groups + QUANT_TABLES + num_groups * pass + group;
            }
        }
    }
    plan.num_groups = num_groups;
    plan.num_dc_groups = num_dc_groups;
    Ok(plan)
}

/// Extract tightly packed local planes in source gi_channel_ order. Call before
/// discarding any globally transformed samples. The mapping remains in plan.
pub fn extract_channels(channels: &[Channel], plan: &StreamPlan) -> Result<Vec<Channel>> {
    let mut result = Vec::new();
    for tile in &plan.channels {
        let c = channels
            .get(tile.channel)
            .ok_or_else(|| Error::InvalidInput("tile channel out of range".into()))?;
        let r = tile.rect;
        if c.width.checked_mul(c.height) != Some(c.samples.len())
            || r.width == 0
            || r.height == 0
            || r.x.checked_add(r.width).is_none_or(|end| end > c.width)
            || r.y.checked_add(r.height).is_none_or(|end| end > c.height)
        {
            return Err(Error::InvalidInput("tile lies outside its channel".into()));
        }
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(r.width * r.height)
            .map_err(|_| Error::Limit("tile samples"))?;
        for y in r.y..r.y + r.height {
            samples.extend_from_slice(&c.samples[y * c.width + r.x..y * c.width + r.x + r.width]);
        }
        result.push(Channel {
            width: r.width,
            height: r.height,
            hshift: c.hshift,
            vshift: c.vshift,
            samples,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(width: usize, height: usize, shift: i32) -> ChannelShape {
        ChannelShape {
            width,
            height,
            hshift: shift,
            vshift: shift,
        }
    }

    #[test]
    fn streaming_areas_and_absolute_dc_ac_ids_match_source() {
        let areas = frame_areas(2305, 2057, 1).unwrap();
        assert_eq!(
            areas,
            [
                Rect {
                    x: 0,
                    y: 0,
                    width: 2048,
                    height: 2048
                },
                Rect {
                    x: 2048,
                    y: 0,
                    width: 257,
                    height: 2048
                },
                Rect {
                    x: 0,
                    y: 2048,
                    width: 2048,
                    height: 9
                },
                Rect {
                    x: 2048,
                    y: 2048,
                    width: 257,
                    height: 9
                },
            ]
        );
        let mut coverage = [0usize; 2];
        let mut seen = vec![false; 4 + 90];
        for area in &areas {
            let shapes = [
                shape(area.width, area.height, 0),
                shape(area.width.div_ceil(8), area.height.div_ceil(8), 3),
            ];
            let plan =
                partition_frame_area(&shapes, 0, 2305, 2057, 1, &Passes::default(), *area).unwrap();
            assert_eq!(
                (plan.global_channels, plan.num_dc_groups, plan.num_groups),
                (0, 4, 90)
            );
            for stream in plan.streams {
                let index = match stream.kind {
                    StreamKind::Dc { group } => {
                        assert_eq!(stream.stream_id, 5 + group);
                        assert!(stream.channels.iter().all(|c| c.channel == 1));
                        group
                    }
                    StreamKind::Ac { group, pass: 0 } => {
                        assert_eq!(stream.stream_id, 30 + group);
                        assert!(stream.channels.iter().all(|c| c.channel == 0));
                        4 + group
                    }
                    _ => panic!("unexpected pass"),
                };
                assert!(!seen[index]);
                seen[index] = true;
                for channel in stream.channels {
                    coverage[channel.channel] += channel.rect.width * channel.rect.height;
                }
            }
        }
        assert!(seen.into_iter().all(|v| v));
        assert_eq!(
            coverage,
            [2305 * 2057, 2305usize.div_ceil(8) * 2057usize.div_ceil(8)]
        );
    }

    #[test]
    fn streaming_tiny_last_area_does_not_keep_small_channels_global() {
        let area = *frame_areas(2049, 2049, 1).unwrap().last().unwrap();
        assert_eq!(
            area,
            Rect {
                x: 2048,
                y: 2048,
                width: 1,
                height: 1
            }
        );
        let plan = partition_frame_area(
            &[shape(1, 1, 0), shape(1, 1, 3)],
            0,
            2049,
            2049,
            1,
            &Passes::default(),
            area,
        )
        .unwrap();
        assert_eq!(plan.global_channels, 0);
        assert_eq!(plan.streams[0].kind, StreamKind::Dc { group: 3 });
        assert_eq!(plan.streams[1].kind, StreamKind::Ac { group: 80, pass: 0 });
        assert_eq!(
            plan.streams[0].channels[0].rect,
            Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1
            }
        );
        assert_eq!(
            plan.streams[1].channels[0].rect,
            Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1
            }
        );
    }

    #[test]
    fn dynamic_dc_offsets_match_decoder_for_every_group_size() {
        for shift in 0..=3 {
            let dim = 1024usize << shift;
            let area = Rect {
                x: dim,
                y: dim,
                width: 1,
                height: 1,
            };
            let plan = partition_frame_area(
                &[shape(1, 1, 0), shape(1, 1, 3)],
                0,
                dim + 1,
                dim + 1,
                shift,
                &Passes::default(),
                area,
            )
            .unwrap();
            assert_eq!(plan.num_dc_groups, 4);
            assert_eq!(plan.num_groups, 81);
            assert_eq!(plan.streams[0].kind, StreamKind::Dc { group: 3 });
            assert_eq!(plan.streams[0].stream_id, 8);
            assert_eq!(plan.streams[0].channels[0].channel, 1);
            assert_eq!(plan.streams[1].kind, StreamKind::Ac { group: 80, pass: 0 });
            assert_eq!(plan.streams[1].stream_id, 110);
            assert_eq!(plan.streams[1].channels[0].channel, 0);
        }
    }

    #[test]
    fn invalid_streaming_areas_reject_before_slicing() {
        for area in [
            Rect {
                x: 1,
                y: 0,
                width: 2048,
                height: 1,
            },
            Rect {
                x: 2048,
                y: 0,
                width: 2,
                height: 1,
            },
            Rect {
                x: usize::MAX,
                y: 0,
                width: 1,
                height: 1,
            },
        ] {
            assert!(
                partition_frame_area(&[shape(1, 1, 0)], 0, 2049, 1, 1, &Passes::default(), area)
                    .is_err()
            );
        }
        assert!(frame_areas(0, 1, 1).is_err());
        assert!(frame_areas(1, 1, 4).is_err());
        assert!(frame_areas(usize::MAX, usize::MAX, 0).is_err());
    }

    fn plane(width: usize, height: usize, seed: i32) -> Channel {
        Channel {
            width,
            height,
            hshift: 0,
            vshift: 0,
            samples: (0..width * height)
                .map(|i| ((i as i32 * 31 + seed) % 1001) - 500)
                .collect(),
        }
    }

    fn undo(channels: &mut Vec<Channel>, parameters: &[SqueezeParams]) {
        for &p in parameters.iter().rev() {
            let offset = if p.in_place {
                p.begin + p.channels
            } else {
                channels.len() - p.channels
            };
            let residuals: Vec<_> = channels.drain(offset..offset + p.channels).collect();
            for (c, residual) in channels[p.begin..p.begin + p.channels]
                .iter_mut()
                .zip(residuals)
            {
                let (width, height) = if p.horizontal {
                    (c.width + residual.width, c.height)
                } else {
                    (c.width, c.height + residual.height)
                };
                let mut samples = vec![0; width * height];
                for y in 0..residual.height {
                    for x in 0..residual.width {
                        let average = c.samples[y * c.width + x];
                        let next = if p.horizontal && x + 1 < c.width {
                            c.samples[y * c.width + x + 1]
                        } else if !p.horizontal && y + 1 < c.height {
                            c.samples[(y + 1) * c.width + x]
                        } else {
                            average
                        };
                        let pos = if p.horizontal {
                            y * width + 2 * x
                        } else {
                            2 * y * width + x
                        };
                        let step = if p.horizontal { 1 } else { width };
                        let previous = if (p.horizontal && x > 0) || (!p.horizontal && y > 0) {
                            samples[pos - step]
                        } else {
                            average
                        };
                        let diff = i64::from(residual.samples[y * residual.width + x])
                            + transforms::smooth_tendency(previous, average, next);
                        let first = i64::from(average) + diff / 2; // inverse truncates toward zero
                        samples[pos] = i32::try_from(first).unwrap();
                        samples[pos + step] = i32::try_from(first - diff).unwrap();
                    }
                }
                if p.horizontal && width % 2 == 1 {
                    for y in 0..height {
                        samples[y * width + width - 1] = c.samples[y * c.width + c.width - 1];
                    }
                }
                if !p.horizontal && height % 2 == 1 {
                    samples[(height - 1) * width..]
                        .copy_from_slice(&c.samples[(c.height - 1) * width..]);
                }
                c.width = width;
                c.height = height;
                c.samples = samples;
                if p.horizontal {
                    c.hshift -= 1;
                } else {
                    c.vshift -= 1;
                }
            }
        }
    }

    #[test]
    fn layout_no_longer_has_private_channel_and_stream_caps() {
        let channels = vec![shape(1, 1, 0); 8193];
        assert!(validate_shapes(&channels, 0).is_ok());
        assert!(squeeze_shapes(&channels, 0, &[]).is_ok());
        // More than a million AC groups, but only ~16K area descriptors.
        let areas = frame_areas(131_200, 131_200, 0).unwrap();
        assert_eq!(areas.len(), 129 * 129);
        assert!(validate_stream_ids(1 << 20, 16_384, 1).is_ok());
        assert!(validate_stream_ids(usize::MAX, 1, 1).is_err());
        assert!(validate_stream_ids(i32::MAX as usize, 1, 1).is_err());
    }

    #[test]
    fn source_default_script_goldens_for_orientation_and_chroma() {
        for (width, height, directions) in [
            (8, 8, ""),
            (9, 8, "H"),
            (8, 9, "V"),
            (9, 9, "VH"),
            (33, 17, "HVHVH"),
            (17, 33, "VHVHV"),
        ] {
            let p = default_squeeze_parameters(&[shape(width, height, 0)], 0).unwrap();
            assert_eq!(
                p.iter()
                    .map(|p| if p.horizontal { 'H' } else { 'V' })
                    .collect::<String>(),
                directions
            );
            assert!(
                p.iter()
                    .all(|p| p.in_place && p.begin == 0 && p.channels == 1)
            );
        }
        let shapes = vec![
            shape(2, 1, -1),
            shape(9, 9, 0),
            shape(9, 9, 0),
            shape(17, 7, 0),
        ];
        let p = default_squeeze_parameters(&shapes, 1).unwrap();
        assert_eq!(
            p,
            [
                SqueezeParams {
                    horizontal: true,
                    in_place: false,
                    begin: 2,
                    channels: 2
                },
                SqueezeParams {
                    horizontal: false,
                    in_place: false,
                    begin: 2,
                    channels: 2
                },
                SqueezeParams {
                    horizontal: false,
                    in_place: true,
                    begin: 1,
                    channels: 3
                },
                SqueezeParams {
                    horizontal: true,
                    in_place: true,
                    begin: 1,
                    channels: 3
                }
            ]
        );
        // Only channel 1 geometry controls the chroma prelude, not channel 2.
        let mut unequal = shapes.clone();
        unequal[2].width = 8;
        assert!(
            default_squeeze_parameters(&unequal, 1)
                .unwrap()
                .iter()
                .all(|p| p.in_place)
        );
        let small = [shape(8, 8, 0); 3];
        let p = default_squeeze_parameters(&small, 0).unwrap();
        let result = squeeze_shapes(&small, 0, &p).unwrap();
        assert_eq!(
            result,
            [
                shape(8, 8, 0),
                shape(4, 4, 1),
                shape(4, 4, 1),
                ChannelShape {
                    width: 4,
                    height: 8,
                    hshift: 1,
                    vshift: 0
                },
                ChannelShape {
                    width: 4,
                    height: 8,
                    hshift: 1,
                    vshift: 0
                },
                shape(4, 4, 1),
                shape(4, 4, 1)
            ]
        );
    }

    #[test]
    fn source_shift_cap_drops_current_and_preceding_parameter() {
        for group_shift in 0..=3 {
            let plan = plan_squeeze(&[shape(1 << 20, 1, 0)], 0, group_shift).unwrap();
            assert!(!plan.use_default);
            assert_eq!(plan.parameters.len(), 6 + usize::from(group_shift));
            assert!(plan.parameters.iter().all(|p| p.horizontal));
        }
        let p = plan_squeeze(&[shape(256, 256, 0); 3], 0, 1).unwrap();
        assert!(p.use_default);
        assert_eq!(
            p.parameters,
            default_squeeze_parameters(&[shape(256, 256, 0); 3], 0).unwrap()
        );
        assert!(plan_squeeze(&[shape(9, 9, 0)], 0, 4).is_err());
    }

    #[test]
    fn append_and_in_place_samples_roundtrip_odd_boundaries_and_meta() {
        for (width, height) in [(1, 1), (1, 17), (17, 1), (9, 9), (17, 33), (33, 17)] {
            let mut original = vec![Channel {
                width: 2,
                height: 1,
                hshift: -1,
                vshift: -1,
                samples: vec![7, 19],
            }];
            original.extend((0..4).map(|i| plane(width, height, i * 17)));
            let mut channels = original.clone();
            let shapes: Vec<_> = channels.iter().map(ChannelShape::from).collect();
            let plan = plan_squeeze(&shapes, 1, 1).unwrap();
            let expected = squeeze_shapes(&shapes, 1, &plan.parameters).unwrap();
            apply_squeeze(&mut channels, 1, &plan.parameters).unwrap();
            assert_eq!(
                channels.iter().map(ChannelShape::from).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(channels[0], original[0]);
            assert_eq!(
                channels.iter().map(|c| c.samples.len()).sum::<usize>(),
                original.iter().map(|c| c.samples.len()).sum::<usize>()
            );
            undo(&mut channels, &plan.parameters);
            assert_eq!(channels, original);
        }
    }

    #[test]
    fn squeeze_errors_are_atomic_and_zero_residuals_are_retained() {
        let original = vec![
            plane(2, 1, 0),
            Channel {
                width: 2,
                height: 1,
                hshift: 0,
                vshift: 0,
                samples: vec![i32::MAX; 2],
            },
        ];
        let mut channels = original.clone();
        let p = SqueezeParams {
            horizontal: true,
            in_place: false,
            begin: 0,
            channels: 2,
        };
        assert!(apply_squeeze(&mut channels, 0, &[p]).is_err());
        assert_eq!(channels, original);
        for p in [
            SqueezeParams { channels: 0, ..p },
            SqueezeParams {
                begin: usize::MAX,
                ..p
            },
            SqueezeParams {
                begin: 1,
                channels: 2,
                ..p
            },
        ] {
            assert!(squeeze_shapes(&[shape(2, 1, 0); 2], 0, &[p]).is_err());
        }
        assert!(squeeze_shapes(&[shape(2, 1, -1), shape(2, 1, 0)], 1, &[p]).is_err());
        let mut c = vec![plane(1, 1, 0)];
        let p = SqueezeParams {
            horizontal: true,
            in_place: true,
            begin: 0,
            channels: 1,
        };
        apply_squeeze(&mut c, 0, &[p]).unwrap();
        assert_eq!((c.len(), c[1].width, c[1].height), (2, 0, 1));
        assert!(c[1].samples.is_empty());
        assert!(squeeze_shapes(&[shape(1, 1, 30)], 0, &[p, p]).is_err());
    }

    #[test]
    fn source_downsampling_brackets_and_empty_passes() {
        assert_eq!(
            Passes::default().bracket(0).unwrap(),
            ShiftRange { min: 0, max: 2 }
        );
        let p = Passes::progressive();
        assert_eq!(
            (0..3).map(|i| p.bracket(i).unwrap()).collect::<Vec<_>>(),
            [
                ShiftRange { min: 2, max: 2 },
                ShiftRange { min: 1, max: 1 },
                ShiftRange { min: 0, max: 0 }
            ]
        );
        let p = Passes {
            num_passes: 5,
            downsample: vec![
                Downsample {
                    factor: 8,
                    last_pass: 0,
                },
                Downsample {
                    factor: 4,
                    last_pass: 2,
                },
                Downsample {
                    factor: 2,
                    last_pass: 3,
                },
            ],
        };
        assert_eq!(
            (0..5).map(|i| p.bracket(i).unwrap()).collect::<Vec<_>>(),
            [
                ShiftRange { min: 3, max: 2 },
                ShiftRange { min: 3, max: 2 },
                ShiftRange { min: 2, max: 2 },
                ShiftRange { min: 1, max: 1 },
                ShiftRange { min: 0, max: 0 }
            ]
        );
        assert!(p.bracket(5).is_err());
        for bad in [
            Passes {
                num_passes: 0,
                downsample: vec![],
            },
            Passes {
                num_passes: 12,
                downsample: vec![],
            },
            Passes {
                num_passes: 2,
                downsample: vec![Downsample {
                    factor: 3,
                    last_pass: 0,
                }],
            },
            Passes {
                num_passes: 2,
                downsample: vec![
                    Downsample {
                        factor: 4,
                        last_pass: 0,
                    },
                    Downsample {
                        factor: 4,
                        last_pass: 1,
                    },
                ],
            },
            Passes {
                num_passes: 3,
                downsample: vec![
                    Downsample {
                        factor: 4,
                        last_pass: 1,
                    },
                    Downsample {
                        factor: 2,
                        last_pass: 1,
                    },
                ],
            },
            Passes {
                num_passes: 11,
                downsample: vec![Downsample {
                    factor: 2,
                    last_pass: 8,
                }],
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn dc_ac_stream_ids_and_prefix_match_source_reference_layout() {
        let shapes = [
            shape(2, 1, -1),
            shape(8, 8, 6),
            shape(513, 257, 3),
            shape(8, 4, 3),
            shape(2050, 1026, 1),
            shape(4099, 2051, 0),
        ];
        let plan = partition_channels(&shapes, 1, 4099, 2051, 1, &Passes::progressive()).unwrap();
        assert_eq!(
            (plan.global_channels, plan.num_dc_groups, plan.num_groups),
            (2, 6, 153)
        );
        assert_eq!(plan.streams.len(), 6 + 153 * 3);
        let dc = &plan.streams[5];
        assert_eq!(dc.kind, StreamKind::Dc { group: 5 });
        assert_eq!(dc.stream_id, 12);
        assert_eq!(
            dc.rect,
            Rect {
                x: 4096,
                y: 2048,
                width: 2048,
                height: 2048
            }
        );
        assert_eq!(
            dc.channels,
            [ChannelTile {
                channel: 2,
                rect: Rect {
                    x: 512,
                    y: 256,
                    width: 1,
                    height: 1
                }
            }]
        );
        // Later small channels do NOT rejoin the global prefix.
        assert!(plan.streams[0].channels.iter().any(|t| t.channel == 3));
        assert!(plan.streams[6].channels.is_empty()); // no shift-2 plane
        let last_half = &plan.streams[6 + 152 * 3 + 1];
        assert_eq!(
            last_half.kind,
            StreamKind::Ac {
                group: 152,
                pass: 1
            }
        );
        assert_eq!(last_half.stream_id, 1 + 3 * 6 + 17 + 153 + 152);
        assert_eq!(
            last_half.channels,
            [ChannelTile {
                channel: 4,
                rect: Rect {
                    x: 2048,
                    y: 1024,
                    width: 2,
                    height: 2
                }
            }]
        );
        assert_eq!(
            plan.streams.last().unwrap().channels,
            [ChannelTile {
                channel: 5,
                rect: Rect {
                    x: 4096,
                    y: 2048,
                    width: 3,
                    height: 3
                }
            }]
        );
    }

    #[test]
    fn shifted_odd_edge_tiles_cover_every_sample_exactly_once() {
        let mut c = plane(129, 65, 3);
        c.hshift = 1;
        c.vshift = 1;
        let channels = vec![c];
        let shapes: Vec<_> = channels.iter().map(ChannelShape::from).collect();
        let plan = partition_channels(&shapes, 0, 257, 129, 0, &Passes::progressive()).unwrap();
        assert_eq!(plan.global_channels, 0);
        let mut reconstructed = vec![0; 129 * 65];
        let mut visits = vec![0u8; 129 * 65];
        for stream in &plan.streams {
            let extracted = extract_channels(&channels, stream).unwrap();
            for (tile, local) in stream.channels.iter().zip(extracted) {
                for y in 0..tile.rect.height {
                    for x in 0..tile.rect.width {
                        let pos = (tile.rect.y + y) * 129 + tile.rect.x + x;
                        visits[pos] += 1;
                        reconstructed[pos] = local.samples[y * local.width + x];
                    }
                }
            }
        }
        assert!(visits.iter().all(|&n| n == 1));
        assert_eq!(reconstructed, channels[0].samples);
        // The last full-resolution group is only 1x1 inside the image, but the
        // shifted channel still contributes 1x1, not floor(1/2)=0.
        assert!(plan.streams.iter().any(|s| s.channels.iter().any(|t| t.rect
            == Rect {
                x: 128,
                y: 64,
                width: 1,
                height: 1
            })));
    }

    #[test]
    fn global_squeeze_then_stream_partition_reassembles_exact_pixels() {
        let original: Vec<_> = (0..3).map(|c| plane(257, 129, c * 47)).collect();
        let mut channels = original.clone();
        let shapes: Vec<_> = channels.iter().map(ChannelShape::from).collect();
        let squeeze = plan_squeeze(&shapes, 0, 0).unwrap();
        apply_squeeze(&mut channels, 0, &squeeze.parameters).unwrap();
        let shapes: Vec<_> = channels.iter().map(ChannelShape::from).collect();
        let layout = partition_channels(&shapes, 0, 257, 129, 0, &Passes::progressive()).unwrap();
        assert!(layout.global_channels > 0 && layout.global_channels < channels.len());
        let mut decoded = channels.clone();
        for c in &mut decoded[layout.global_channels..] {
            c.samples.fill(0);
        }
        let mut visits: Vec<Vec<u8>> = decoded.iter().map(|c| vec![0; c.samples.len()]).collect();
        for stream in &layout.streams {
            for (tile, local) in stream
                .channels
                .iter()
                .zip(extract_channels(&channels, stream).unwrap())
            {
                let c = &mut decoded[tile.channel];
                for y in 0..tile.rect.height {
                    for x in 0..tile.rect.width {
                        let pos = (tile.rect.y + y) * c.width + tile.rect.x + x;
                        visits[tile.channel][pos] += 1;
                        c.samples[pos] = local.samples[y * local.width + x];
                    }
                }
            }
        }
        for v in &visits[layout.global_channels..] {
            assert!(v.iter().all(|&n| n == 1));
        }
        assert_eq!(decoded, channels);
        undo(&mut decoded, &squeeze.parameters);
        assert_eq!(decoded, original);
    }

    #[test]
    fn reject_uncovered_shifted_layouts_and_malformed_extraction() {
        let mut thin = shape(1, 257, 0);
        thin.hshift = 8;
        // AC tiles have zero width at hshift=8 for 128px groups, despite
        // min(hshift,vshift)=0. Source's squeeze cap prevents this situation.
        assert!(partition_channels(&[thin], 0, 256, 257, 0, &Passes::default()).is_err());
        assert!(partition_channels(&[shape(300, 1, 0)], 0, 257, 1, 0, &Passes::default()).is_err());
        assert!(partition_channels(&[], 0, usize::MAX, usize::MAX, 0, &Passes::default()).is_err());
        let shapes = [shape(8, 8, 0); 3];
        let plan = partition_channels(&shapes, 0, 8, 8, 1, &Passes::default()).unwrap();
        assert_eq!(plan.global_channels, 3);
        assert!(plan.streams.iter().all(|s| s.channels.is_empty()));
        let mut malformed = plan.streams[0].clone();
        malformed.channels.push(ChannelTile {
            channel: 0,
            rect: Rect {
                x: usize::MAX,
                y: 0,
                width: 1,
                height: 1,
            },
        });
        assert!(extract_channels(&[plane(8, 8, 0)], &malformed).is_err());
    }
}
