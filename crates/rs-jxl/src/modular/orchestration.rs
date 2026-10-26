// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Full-frame lossless Modular orchestration from enc_modular.cc:
//! ComputeEncodingData -> PrepareStreamParams -> EncodeStream. Geometry and
//! default squeeze scheduling are in layout.rs; palette policy in palette.rs.
//! DCGlobal retains the small-channel prefix, DC takes min-shift >=3, and AC
//! passes take the frame's downsampling brackets. No group-local approximation
//! of global squeeze, no regenerated WP search during entropy serialization.
//!
//! RCT/WP candidate counts supplied in ModularOptions are explicit overrides of
//! source speed-profile counts, not proxy costs. RCT stage, speed, responsive,
//! direct-audit and headroom gates still apply. Automatic RCT is requested by
//! rct_candidates; otherwise rct is an explicit global transform (None disables).
//! Samples are already normalized signed Modular words, including floating-point
//! bit patterns. Groupwise encoding follows enc_frame.cc::EncodeFrameStreaming:
//! DC-sized areas, no global palette/squeeze, independently coded local streams.
//! region_input supplies borrowed native samples to this same area preparation
//! path and emits completed sections. Parent owns frame headers, TOC and sinks.
//! Lossy quantization and within-section incremental encoding are unsupported.

#![forbid(unsafe_code)]

use super::candidate_cache::{PreparationKey, prepared_bytes};
use super::cost::{self, ChannelView};
use super::encode::receipt::RepresentationReceipt;
use super::encode::{self, Group, ModularOptions, Prepared};
use super::group::{self, Channel, Transform};
use super::layout::{self, ChannelShape, PartitionPlan, Passes, StreamKind};
use super::palette::{self, PaletteMode, PaletteOptions, PaletteStage, PaletteState};
use super::{frame, weighted};
use crate::encode::budget::{self, SectionBudget, TrialEncode, complete_or_return};
use crate::execution;
use crate::region_input::ImageDescriptor;
use crate::{Error, Image, Result, tools::patches::PatchDictionary};
use std::sync::Arc;

pub(super) struct PreparedLayout {
    pub(super) global: Prepared,
    pub(super) streams: Vec<Prepared>,
    pub(super) partition: PartitionPlan,
}

fn reuse_preparation(
    image: &Image,
    options: &ModularOptions,
    key: impl FnOnce() -> PreparationKey,
    build: impl FnOnce() -> Result<PreparedLayout>,
) -> Result<Arc<PreparedLayout>> {
    if let Some(cache) = &options.candidate_cache
        && options.representation_receipt.is_none()
        && cache.matches(image)
    {
        return cache.get_or_build(key(), build, prepared_bytes);
    }
    build().map(Arc::new)
}

fn source_policy(options: &ModularOptions, responsive: bool) -> Result<PaletteOptions> {
    let mut policy = options.palette_policy.clone().unwrap_or(PaletteOptions {
        palette_colors: 0,
        channel_colors_pre_transform_percent: 0.0,
        channel_colors_percent: 0.0,
        ..PaletteOptions::default()
    });
    policy.responsive |= responsive;
    policy.validate()?;
    if policy.responsive && !options.squeeze.is_empty() {
        return Err(Error::InvalidInput(
            "responsive global squeeze cannot be combined with explicit local squeeze passes"
                .into(),
        ));
    }
    Ok(policy)
}

/// Full-frame fallback reasons. LocalSqueeze is the Rust explicit-pass
/// compatibility route; the others are source CanDoStreamingEncoding gates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupwiseFallback {
    NotRequested,
    AtMostEightGroups,
    PatchDictionary,
    CustomPasses,
    Responsive,
    LocalSqueeze,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupwiseDecision {
    FrameAreas,
    FullFrame(GroupwiseFallback),
}

/// Source streaming eligibility within this API's pure pixel-lossless domain.
/// `palette_policy.groupwise=true` is an explicit buffering=2 request, not the
/// buffering=-1 speed heuristic or buffering=1 2048-pixel threshold. Unsupported
/// JPEG/VarDCT, resampling, noise and DC-frame inputs are not represented here.
/// Source-rejected requests use the ordinary full-frame route, not partial
/// groupwise gates. `has_patches` means an explicitly supplied dictionary.
pub fn groupwise_decision(
    image: &Image,
    options: &ModularOptions,
    group_size_shift: u8,
    passes: &Passes,
    responsive: bool,
    has_patches: bool,
) -> Result<GroupwiseDecision> {
    groupwise_decision_geometry(
        image.width(),
        image.height(),
        options,
        group_size_shift,
        passes,
        responsive,
        has_patches,
    )
}

pub(crate) fn groupwise_decision_geometry(
    width: u32,
    height: u32,
    options: &ModularOptions,
    group_size_shift: u8,
    passes: &Passes,
    responsive: bool,
    has_patches: bool,
) -> Result<GroupwiseDecision> {
    options.validate()?;
    passes.validate()?;
    if group_size_shift > 3 {
        return Err(Error::InvalidInput("group size shift must be 0..3".into()));
    }
    let policy = source_policy(options, responsive)?;
    let fallback = |reason| Ok(GroupwiseDecision::FullFrame(reason));
    if !policy.groupwise {
        return fallback(GroupwiseFallback::NotRequested);
    }
    let dim = 128usize << group_size_shift;
    let groups = (width as usize)
        .div_ceil(dim)
        .checked_mul((height as usize).div_ceil(dim))
        .ok_or(Error::Limit("AC group count"))?;
    if groups <= 8 {
        return fallback(GroupwiseFallback::AtMostEightGroups);
    }
    if has_patches {
        return fallback(GroupwiseFallback::PatchDictionary);
    }
    if passes != &Passes::default() {
        return fallback(GroupwiseFallback::CustomPasses);
    }
    if policy.responsive {
        return fallback(GroupwiseFallback::Responsive);
    }
    if !options.squeeze.is_empty() {
        return fallback(GroupwiseFallback::LocalSqueeze);
    }
    Ok(GroupwiseDecision::FrameAreas)
}

fn effective_depth(image: &Image) -> Result<PaletteState> {
    effective_depth_metadata(image.metadata())
}

fn effective_depth_metadata(metadata: &crate::metadata::ImageMetadata) -> Result<PaletteState> {
    // enc_modular.cc: integer samples have an implicit zero sign bit; floating
    // samples already include it. Image::modular_sample owns all conversion.
    let mut state = PaletteState {
        meta_channels: 0,
        max_bitdepth: 0,
        max_value: if metadata.bit_depth.bits_per_sample < 32 {
            ((1u32 << metadata.bit_depth.bits_per_sample) - 1) as i32
        } else {
            0
        },
    };
    for depth in std::iter::once(metadata.bit_depth)
        .chain(metadata.extra_channels.iter().map(|extra| extra.bit_depth))
    {
        depth.validate()?;
        state.max_bitdepth = state
            .max_bitdepth
            .max(i32::from(depth.bits_per_sample) + i32::from(!depth.is_float()));
    }
    Ok(state)
}

fn raw_options(options: &ModularOptions) -> ModularOptions {
    let mut raw = options.clone();
    raw.rct = None;
    raw.rct_candidates = None;
    raw.palette_colors = None;
    raw.palette_predictor = None;
    raw.palette_policy = None;
    raw.squeeze.clear();
    raw
}

fn apply_global_rct(
    channels: &mut [Channel],
    state: &mut PaletteState,
    options: &ModularOptions,
    policy: &PaletteOptions,
    max_depth: i32,
) -> Result<Vec<Transform>> {
    let global_rct = if options.rct_candidates.is_some() {
        // Init sets automatic colorspace to YCoCg for responsive lossless.
        if policy.responsive || policy.speed_tier > 5 {
            Some(6)
        } else {
            None
        }
    } else {
        options.rct
    };
    let mut transforms = Vec::new();
    if policy.do_color
        && state.max_bitdepth + 1 < max_depth
        && equal_color_range(channels, state.meta_channels)
        && let Some(kind) = global_rct
        && let Some(t) = group::apply_rct(channels, state.meta_channels, state.meta_channels, kind)?
    {
        transforms.push(t);
        state.max_bitdepth += 1;
    }
    Ok(transforms)
}

fn views(channels: &[Channel]) -> Vec<ChannelView<'_>> {
    channels
        .iter()
        .map(|c| ChannelView {
            pixels: &c.samples,
            width: c.width,
            height: c.height,
            stride: c.width,
        })
        .collect()
}

fn equal_color_range(channels: &[Channel], begin: usize) -> bool {
    let Some(end) = begin.checked_add(3).filter(|&end| end <= channels.len()) else {
        return false;
    };
    let shape = ChannelShape::from(&channels[begin]);
    channels[begin..end]
        .iter()
        .all(|c| ChannelShape::from(c) == shape)
}

fn choose_wp(
    channels: &[Channel],
    options: &ModularOptions,
    policy: &PaletteOptions,
    can_search: bool,
    source_profile: bool,
) -> Result<weighted::Header> {
    if !can_search
        || !options.predictor.has_weighted()
        || channels.iter().all(|c| c.samples.is_empty())
    {
        return weighted::Header::preset(u32::from(options.weighted_preset));
    }
    let mut local = options.clone();
    if source_profile && local.weighted_candidates.is_none() {
        local.weighted_candidates = match policy.speed_tier {
            -1..=1 => Some(5),
            2 => Some(2),
            _ => None,
        };
    }
    // Explicit presets remain explicit; only an actual multi-mode search
    // violates the direct-audit route.
    if policy.direct && local.weighted_candidates.is_some_and(|n| n > 1) {
        return Err(Error::InvalidInput(
            "direct palette policy cannot reach WP search".into(),
        ));
    }
    encode::weighted_header(channels, &local)
}

#[allow(clippy::too_many_arguments)]
fn finish_stream(
    mut channels: Vec<Channel>,
    mut transforms: Vec<Transform>,
    mut state: PaletteState,
    stream_id: usize,
    local: bool,
    allow_rct: bool,
    can_search: bool,
    bits: u8,
    options: &ModularOptions,
    policy: &PaletteOptions,
) -> Result<Prepared> {
    if channels.is_empty() {
        return Ok(Prepared {
            planes: channels,
            transforms,
            wp: weighted::Header::default(),
            stream_id,
        });
    }
    if local && options.palette_policy.is_some() {
        transforms.extend(palette::apply_palettes(
            &mut channels,
            &mut state,
            bits,
            PaletteStage::PostRct,
            policy,
        )?);
    }
    if can_search
        && allow_rct
        && !policy.responsive
        && policy.speed_tier <= 5
        && policy.do_color
        && equal_color_range(&channels, state.meta_channels)
        && let Some(count) = options.rct_candidates
    {
        if policy.direct {
            return Err(Error::InvalidInput(
                "direct palette policy cannot reach RCT search".into(),
            ));
        }
        let selected = cost::select_rct(&views(&channels), state.meta_channels, count)?;
        if let Some(t) = group::apply_rct(
            &mut channels,
            state.meta_channels,
            state.meta_channels,
            selected.rct_type,
        )? {
            transforms.push(t);
        }
    }
    // Existing explicit passes are a user override, separate from source
    // responsive global squeeze. Never apply them to palette/meta tables.
    for &horizontal in &options.squeeze {
        if state.meta_channels < channels.len() {
            transforms.extend(group::apply_squeeze(
                &mut channels,
                state.meta_channels,
                horizontal,
            )?);
        }
    }
    if transforms.len() > 273 {
        return Err(Error::Limit("Modular transform count"));
    }
    let wp = choose_wp(
        &channels,
        options,
        policy,
        can_search,
        options.palette_policy.is_some(),
    )?;
    Ok(Prepared {
        planes: channels,
        transforms,
        wp,
        stream_id,
    })
}

fn prepare_layout(
    image: &Image,
    options: &ModularOptions,
    group_size_shift: u8,
    passes: &Passes,
    responsive: bool,
) -> Result<PreparedLayout> {
    if let Some(collector) = &options.representation_receipt {
        collector.begin()?;
    }
    options.validate()?;
    passes.validate()?;
    if group_size_shift > 3 {
        return Err(Error::InvalidInput("group size shift must be 0..3".into()));
    }
    let policy = source_policy(options, responsive)?;
    let mut state = effective_depth(image)?;
    // The 16-bit promise is stricter than Level 10's 32-bit headroom. Respect
    // it even when the container independently needs Level 10 for other fields.
    let max_depth = if image.metadata().modular_16_bit_buffer_sufficient {
        16
    } else {
        32
    };
    if state.max_bitdepth > max_depth {
        return Err(Error::InvalidInput(
            "sample depth exceeds promised Modular buffer".into(),
        ));
    }
    let raw = raw_options(options);
    let full = Group {
        x: 0,
        y: 0,
        width: image.width() as usize,
        height: image.height() as usize,
        stream_id: 0,
    };
    let (mut channels, mut transforms) = encode::planes(image, full, &raw)?;
    group::validate_layout(&channels, 0)?;
    if options.palette_policy.is_some() {
        transforms.extend(palette::apply_palettes(
            &mut channels,
            &mut state,
            image.bits(),
            PaletteStage::PreRct,
            &policy,
        )?);
    } else if !policy.responsive
        && let Some(limit) = options.palette_colors
    {
        let count = channels.len();
        let begin = state.meta_channels;
        let mode = options
            .palette_predictor
            .map_or(PaletteMode::Absolute, PaletteMode::ExactDelta);
        if let Some(t) = group::apply_palette_range(
            &mut channels,
            &mut state.meta_channels,
            image.bits(),
            begin,
            count,
            limit,
            true,
            mode,
        )? {
            transforms.push(t);
        }
    }
    let palette_transform_count = transforms.len();
    transforms.extend(apply_global_rct(
        &mut channels,
        &mut state,
        options,
        &policy,
        max_depth,
    )?);
    if policy.responsive && !channels.is_empty() && state.max_bitdepth + 2 < max_depth {
        let shapes: Vec<_> = channels.iter().map(ChannelShape::from).collect();
        let squeeze = layout::plan_squeeze(&shapes, state.meta_channels, group_size_shift)?;
        if !squeeze.parameters.is_empty() {
            layout::apply_squeeze(&mut channels, state.meta_channels, &squeeze.parameters)?;
            transforms.push(Transform::SqueezeScript(if squeeze.use_default {
                Vec::new()
            } else {
                squeeze.parameters
            }));
        }
        // Source increments even if a tiny grayscale default schedule is empty.
        state.max_bitdepth += 2;
    }
    let allow_rct = state.max_bitdepth < max_depth;
    let receipt = options
        .representation_receipt
        .as_ref()
        .map(|_| {
            RepresentationReceipt::full_image(
                image.width(),
                image.height(),
                image.channel_count(),
                &transforms[..palette_transform_count],
                channels.len() - state.meta_channels,
                state.meta_channels,
            )
        })
        .transpose()?;
    let shapes: Vec<_> = channels.iter().map(ChannelShape::from).collect();
    let partition = layout::partition_channels(
        &shapes,
        state.meta_channels,
        full.width,
        full.height,
        group_size_shift,
        passes,
    )?;
    let streams = execution::try_map(&partition.streams, |_, stream| {
        let local = layout::extract_channels(&channels, stream)?;
        // PrepareStreamParams initializes these unused local policy values to 0.
        let state = PaletteState {
            meta_channels: 0,
            max_bitdepth: 0,
            max_value: 0,
        };
        finish_stream(
            local,
            Vec::new(),
            state,
            stream.stream_id,
            true,
            allow_rct,
            true,
            image.bits(),
            options,
            &policy,
        )
    })?;
    channels.truncate(partition.global_channels);
    let can_search = partition.num_groups == 1 && passes.num_passes == 1;
    let global = if can_search {
        finish_stream(
            channels,
            transforms,
            state,
            0,
            false,
            allow_rct,
            true,
            image.bits(),
            options,
            &policy,
        )?
    } else {
        // Source never runs local transforms/search on global stream 0 when
        // other streams exist; its complete transform script must still emit.
        let wp = choose_wp(
            &channels,
            options,
            &policy,
            false,
            options.palette_policy.is_some(),
        )?;
        Prepared {
            planes: channels,
            transforms,
            wp,
            stream_id: 0,
        }
    };
    if global.transforms.len() > 273 {
        return Err(Error::Limit("Modular global transform count"));
    }
    if let (Some(collector), Some(receipt)) = (&options.representation_receipt, receipt) {
        collector.publish(receipt)?;
    }
    Ok(PreparedLayout {
        global,
        streams,
        partition,
    })
}

#[path = "group_session.rs"]
pub(crate) mod session;

fn read_image_area(
    image: &Image,
    options: &ModularOptions,
    area: layout::Rect,
) -> Result<Vec<Channel>> {
    let raw = raw_options(options);
    let (channels, _) = encode::planes(
        image,
        Group {
            x: area.x,
            y: area.y,
            width: area.width,
            height: area.height,
            stream_id: 0,
        },
        &raw,
    )?;
    Ok(channels)
}

#[cfg(test)]
fn prepare_frame_area(
    image: &Image,
    options: &ModularOptions,
    group_size_shift: u8,
    area: layout::Rect,
) -> Result<PreparedLayout> {
    prepare_frame_area_channels(
        &ImageDescriptor::from_image(image)?,
        options,
        group_size_shift,
        area,
        read_image_area(image, options, area)?,
    )
}

fn prepare_frame_area_channels(
    descriptor: &ImageDescriptor,
    options: &ModularOptions,
    group_size_shift: u8,
    area: layout::Rect,
    mut channels: Vec<Channel>,
) -> Result<PreparedLayout> {
    let _scope = crate::profile::scope("modular.prepare_frame_area");
    let policy = source_policy(options, false)?;
    let mut state = effective_depth_metadata(descriptor.metadata())?;
    let max_depth = if descriptor.metadata().modular_16_bit_buffer_sufficient {
        16
    } else {
        32
    };
    if state.max_bitdepth > max_depth {
        return Err(Error::InvalidInput(
            "sample depth exceeds promised Modular buffer".into(),
        ));
    }
    group::validate_layout(&channels, 0)?;
    // Source groupwise gates skip global palettes and global squeeze, but not
    // an explicit/fast-tier global RCT. Reapply that same script to every area.
    let transforms = apply_global_rct(&mut channels, &mut state, options, &policy, max_depth)?;
    let shapes: Vec<_> = channels.iter().map(ChannelShape::from).collect();
    let partition = layout::partition_frame_area(
        &shapes,
        0,
        descriptor.width() as usize,
        descriptor.height() as usize,
        group_size_shift,
        &Passes::default(),
        area,
    )?;
    let allow_rct = state.max_bitdepth < max_depth;
    let streams = execution::try_map(&partition.streams, |_, stream| {
        let channels = layout::extract_channels(&channels, stream)?;
        finish_stream(
            channels,
            Vec::new(),
            PaletteState {
                meta_channels: 0,
                max_bitdepth: 0,
                max_value: 0,
            },
            stream.stream_id,
            true,
            allow_rct,
            true,
            descriptor.bits(),
            options,
            &policy,
        )
    })?;
    // The first streaming area contains a large full-resolution color channel,
    // so DCGlobal has a transform header, but no non-meta pixel payload.
    let global = Prepared {
        planes: Vec::new(),
        transforms,
        wp: weighted::Header::preset(u32::from(options.weighted_preset))?,
        stream_id: 0,
    };
    Ok(PreparedLayout {
        global,
        streams,
        partition,
    })
}

pub(crate) struct EncodedRegionArea {
    pub transforms: Vec<Transform>,
    /// Canonical TOC indices; bounded to this area's sections only.
    pub sections: Vec<(usize, Vec<u8>)>,
}

/// Called under the caller's explicit pool or null-runner scope. Input leases
/// and output callbacks never enter this owned-data work item.
pub(crate) fn encode_region_area(
    descriptor: &ImageDescriptor,
    channels: Vec<Channel>,
    options: &ModularOptions,
    group_size_shift: u8,
    area: layout::Rect,
    first: bool,
    limit: usize,
) -> Result<EncodedRegionArea> {
    let PreparedLayout {
        global,
        streams,
        partition,
    } = prepare_frame_area_channels(descriptor, options, group_size_shift, area, channels)?;
    let mut sections = Vec::new();
    sections
        .try_reserve_exact(streams.len() + usize::from(first))
        .map_err(|_| Error::Limit("region encoded sections"))?;
    if first {
        sections.push((
            0,
            frame::encode_prepared_independent(
                &global,
                options,
                limit,
                true,
                None,
                descriptor.bits(),
            )?,
        ));
    }
    let encoded = execution::try_map(&streams, |_, stream| {
        frame::encode_prepared_independent(stream, options, limit, false, None, descriptor.bits())
    })?;
    for (bytes, stream) in encoded.into_iter().zip(&partition.streams) {
        sections.push((section_index(stream.kind, &partition), bytes));
    }
    Ok(EncodedRegionArea {
        transforms: global.transforms,
        sections,
    })
}

fn section_index(kind: StreamKind, partition: &PartitionPlan) -> usize {
    match kind {
        StreamKind::Dc { group } => 1 + group,
        StreamKind::Ac { group, pass } => {
            2 + partition.num_dc_groups + partition.num_groups * pass + group
        }
    }
}

fn insert_section(
    sections: &mut [Option<Vec<u8>>],
    index: usize,
    bytes: Vec<u8>,
    size: &mut usize,
    limit: usize,
) -> Result<()> {
    let slot = sections
        .get_mut(index)
        .ok_or_else(|| Error::Encoding("streaming TOC index out of range".into()))?;
    if slot.is_some() {
        return Err(Error::Encoding("streaming section encoded twice".into()));
    }
    *size = size
        .checked_add(bytes.len())
        .filter(|&n| n <= limit)
        .ok_or(Error::Limit("frame section bytes"))?;
    *slot = Some(bytes);
    Ok(())
}

fn encode_frame_areas(
    image: &Image,
    options: &ModularOptions,
    group_size_shift: u8,
    limit: usize,
    budget: Option<&SectionBudget>,
) -> Result<TrialEncode<Vec<Vec<u8>>>> {
    let _scope = crate::profile::scope("modular.frame_areas");
    if let Some(collector) = &options.representation_receipt {
        collector.begin()?;
    }
    let descriptor = ImageDescriptor::from_image(image)?;
    let areas = layout::frame_areas(
        image.width() as usize,
        image.height() as usize,
        group_size_shift,
    )?;
    let mut local = options.clone();
    // HistogramParams::ForModular(..., streaming_mode=true), also for local
    // ModularGenericCompress. No ComputeTree/ComputeTokens shared-frame stage.
    local.entropy_options.streaming_mode = true;
    local.tree_groups = None;
    local.tree_group_splits.clear();
    let mut sections: Vec<Option<Vec<u8>>> = Vec::new();
    let mut script: Option<Vec<Transform>> = None;
    let mut size = 0;
    for area in areas {
        if let Some(lower_bound) = budget.and_then(SectionBudget::reached) {
            return Ok(TrialEncode::Bounded { lower_bound });
        }
        #[cfg(test)]
        if let Some(budget) = budget {
            budget
                .areas_started
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let prepared = reuse_preparation(
            image,
            &local,
            || PreparationKey::area(&local, group_size_shift, area),
            || {
                prepare_frame_area_channels(
                    &descriptor,
                    &local,
                    group_size_shift,
                    area,
                    read_image_area(image, &local, area)?,
                )
            },
        )?;
        let PreparedLayout {
            global,
            streams,
            partition,
        } = prepared.as_ref();
        if let Some(script) = &script {
            if script != &global.transforms {
                return Err(Error::Encoding(
                    "streaming global transform script changed between areas".into(),
                ));
            }
        } else {
            if let Some(collector) = &options.representation_receipt {
                collector.publish(RepresentationReceipt::groupwise(
                    image.width(),
                    image.height(),
                    image.channel_count(),
                ))?;
            }
            sections.resize_with(partition.num_dc_groups + partition.num_groups + 2, || None);
            script = Some(global.transforms.clone());
            let bytes = complete_or_return!(budget::section(budget, || {
                frame::encode_prepared_independent(global, &local, limit, true, None, image.bits())
            }));
            insert_section(&mut sections, 0, bytes, &mut size, limit)?;
            insert_section(
                &mut sections,
                1 + partition.num_dc_groups,
                Vec::new(),
                &mut size,
                limit,
            )?;
        }
        let encoded = complete_or_return!(budget::sections(budget, streams, |_, stream| {
            frame::encode_prepared_independent(stream, &local, limit, false, None, image.bits())
        }));
        for (bytes, stream) in encoded.into_iter().zip(&partition.streams) {
            insert_section(
                &mut sections,
                section_index(stream.kind, partition),
                bytes,
                &mut size,
                limit,
            )?;
        }
        // The optional scoped cache can retain immutable preparation within its
        // byte cap. MA samples, trees and entropy models remain candidate-local.
    }
    let sections = sections
        .into_iter()
        .map(|section| {
            section.ok_or_else(|| Error::Encoding("streaming TOC section missing".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TrialEncode::Complete(sections))
}

fn toc_sections(
    global: Vec<u8>,
    encoded: Vec<Vec<u8>>,
    partition: &PartitionPlan,
    passes: &Passes,
    limit: usize,
) -> Result<Vec<Vec<u8>>> {
    if encoded.len() != partition.streams.len() {
        return Err(Error::Encoding(
            "encoded stream/layout count mismatch".into(),
        ));
    }
    let ac_count = partition
        .num_groups
        .checked_mul(usize::from(passes.num_passes))
        .ok_or(Error::Limit("TOC AC sections"))?;
    let count = partition
        .num_dc_groups
        .checked_add(ac_count)
        .and_then(|n| n.checked_add(2))
        .ok_or(Error::Limit("TOC sections"))?;
    let mut sections = vec![Vec::new(); count];
    sections[0] = global;
    for (bytes, stream) in encoded.into_iter().zip(&partition.streams) {
        let index = section_index(stream.kind, partition);
        sections[index] = bytes;
    }
    let size = sections.iter().try_fold(0usize, |n, s| {
        n.checked_add(s.len())
            .ok_or(Error::Limit("frame section bytes"))
    })?;
    if size > limit {
        return Err(Error::Limit("frame section bytes"));
    }
    if partition.num_groups == 1 && passes.num_passes == 1 {
        // All channels fit DCGlobal in this case. Other sections are empty;
        // keeping the concatenation explicit documents source TOC collapse.
        let mut single = Vec::new();
        single
            .try_reserve_exact(size)
            .map_err(|_| Error::Limit("combined section bytes"))?;
        for section in sections {
            single.extend(section);
        }
        return Ok(vec![single]);
    }
    Ok(sections)
}

/// Complete byte-aligned sections in TOC order: global, DC groups, empty
/// ACGlobal, then pass-major AC groups. One group + one pass collapses to one
/// section. `passes` must also be written unchanged to the frame header.
/// Uses the core execution scope: ambient pool normally, caller thread for a
/// null-runner candidate. No stream changes its transform or search policy.
/// A qualified groupwise request follows source frame-area streaming and uses
/// independent local trees regardless of `shared` or `options.tree_groups`.
/// Rejected source eligibility
/// falls back to full-frame encoding with `shared`; inspect groupwise_decision.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_layout(
    image: &Image,
    options: &ModularOptions,
    group_size_shift: u8,
    passes: &Passes,
    responsive: bool,
    shared: bool,
    limit: usize,
    patches: Option<&PatchDictionary>,
) -> Result<Vec<Vec<u8>>> {
    encode_layout_trial(
        image,
        options,
        group_size_shift,
        passes,
        responsive,
        shared,
        limit,
        patches,
        None,
    )?
    .into_complete()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_layout_trial(
    image: &Image,
    options: &ModularOptions,
    group_size_shift: u8,
    passes: &Passes,
    responsive: bool,
    shared: bool,
    limit: usize,
    patches: Option<&PatchDictionary>,
    budget: Option<&SectionBudget>,
) -> Result<TrialEncode<Vec<Vec<u8>>>> {
    let _scope = crate::profile::scope("modular.layout");
    if groupwise_decision(
        image,
        options,
        group_size_shift,
        passes,
        responsive,
        patches.is_some(),
    )? == GroupwiseDecision::FrameAreas
    {
        return encode_frame_areas(image, options, group_size_shift, limit, budget);
    }
    let mut full_options;
    let options = if options.palette_policy.as_ref().is_some_and(|p| p.groupwise) {
        full_options = options.clone();
        full_options.palette_policy.as_mut().unwrap().groupwise = false;
        &full_options
    } else {
        options
    };
    let prepared = {
        let _scope = crate::profile::scope("modular.prepare_layout");
        reuse_preparation(
            image,
            options,
            || PreparationKey::full(options, group_size_shift, passes, responsive),
            || prepare_layout(image, options, group_size_shift, passes, responsive),
        )?
    };
    let PreparedLayout {
        global,
        streams,
        partition,
    } = prepared.as_ref();
    // Source EncodeGroups excludes non-streaming one-group/one-pass frames:
    // their sections share a collapsed writer, so keep their outcome exact.
    let budget = budget.filter(|_| partition.num_groups != 1 || passes.num_passes != 1);
    let (global, encoded) = if shared {
        complete_or_return!(frame::encode_prepared_shared_borrowed_trial(
            Some(global),
            streams,
            options,
            limit,
            patches,
            image.bits(),
            budget,
        ))
    } else {
        let global = complete_or_return!(budget::section(budget, || {
            frame::encode_prepared_independent(global, options, limit, true, patches, image.bits())
        }));
        let encoded = complete_or_return!(budget::sections(budget, streams, |_, g| {
            frame::encode_prepared_independent(g, options, limit, false, None, image.bits())
        }));
        (global, encoded)
    };
    Ok(TrialEncode::Complete(toc_sections(
        global, encoded, partition, passes, limit,
    )?))
}

#[cfg(test)]
#[path = "receipt_tests.rs"]
mod receipt_tests;

#[cfg(test)]
mod tests {
    use super::super::{encode::FixedTree, predictor::Predictor};
    use super::*;
    use crate::{
        PixelData, PixelLayout,
        metadata::{BitDepth, ExtraChannelInfo, ExtraChannelType},
    };

    fn options() -> ModularOptions {
        ModularOptions {
            predictor: Predictor::Gradient,
            fixed_tree: Some(FixedTree::GradientLeaf),
            lz77: false,
            ..ModularOptions::default()
        }
    }

    fn streaming_options() -> ModularOptions {
        ModularOptions {
            rct: None,
            palette_policy: Some(PaletteOptions {
                groupwise: true,
                palette_colors: 0,
                channel_colors_percent: 0.0,
                channel_colors_pre_transform_percent: 0.0,
                ..PaletteOptions::default()
            }),
            ..options()
        }
    }

    fn gray(width: u32, height: u32) -> Image {
        Image::new(
            width,
            height,
            PixelLayout::Gray,
            PixelData::U8(
                (0..width as usize * height as usize)
                    .map(|i| (i * 17) as u8)
                    .collect(),
            ),
        )
        .unwrap()
    }

    #[test]
    fn source_groupwise_eligibility_and_explicit_buffering_semantics() {
        let image = gray(1025, 1);
        let mut options = streaming_options();
        let passes = Passes::default();
        assert_eq!(
            groupwise_decision(&image, &options, 0, &passes, false, false).unwrap(),
            GroupwiseDecision::FrameAreas
        );
        // A forced buffering=2 request does not use the automatic speed gate.
        options.palette_policy.as_mut().unwrap().speed_tier = -1;
        assert_eq!(
            groupwise_decision(&image, &options, 0, &passes, false, false).unwrap(),
            GroupwiseDecision::FrameAreas
        );
        for (small, shift, passes, responsive, patches, reason) in [
            (
                true,
                0,
                Passes::default(),
                false,
                false,
                GroupwiseFallback::AtMostEightGroups,
            ),
            (
                false,
                1,
                Passes::default(),
                false,
                false,
                GroupwiseFallback::AtMostEightGroups,
            ),
            (
                false,
                0,
                Passes::default(),
                false,
                true,
                GroupwiseFallback::PatchDictionary,
            ),
            (
                false,
                0,
                Passes::progressive(),
                false,
                false,
                GroupwiseFallback::CustomPasses,
            ),
            (
                false,
                0,
                Passes::default(),
                true,
                false,
                GroupwiseFallback::Responsive,
            ),
        ] {
            let small_image = gray(1024, 1);
            let image = if small { &small_image } else { &image };
            assert_eq!(
                groupwise_decision(image, &options, shift, &passes, responsive, patches).unwrap(),
                GroupwiseDecision::FullFrame(reason)
            );
        }
        options.palette_policy.as_mut().unwrap().responsive = true;
        assert_eq!(
            groupwise_decision(&image, &options, 0, &passes, false, false).unwrap(),
            GroupwiseDecision::FullFrame(GroupwiseFallback::Responsive)
        );
        options.palette_policy.as_mut().unwrap().responsive = false;
        options.squeeze.push(true);
        assert_eq!(
            groupwise_decision(&image, &options, 0, &passes, false, false).unwrap(),
            GroupwiseDecision::FullFrame(GroupwiseFallback::LocalSqueeze)
        );
        options.squeeze.clear();
        options.palette_policy.as_mut().unwrap().groupwise = false;
        assert_eq!(
            groupwise_decision(&image, &options, 0, &passes, false, false).unwrap(),
            GroupwiseDecision::FullFrame(GroupwiseFallback::NotRequested)
        );
    }

    #[test]
    fn source_declined_streaming_uses_complete_full_frame_policy() {
        for (width, passes, responsive, local_squeeze) in [
            (17, Passes::default(), false, false),
            (1025, Passes::default(), true, false),
            (1025, Passes::progressive(), false, false),
            (1025, Passes::default(), false, true),
        ] {
            let image = gray(width, 1);
            let mut options = streaming_options();
            if local_squeeze {
                options.squeeze.push(true);
            }
            let mut full = options.clone();
            full.palette_policy.as_mut().unwrap().groupwise = false;
            let expected =
                encode_layout(&image, &full, 0, &passes, responsive, false, 1 << 20, None).unwrap();
            let actual = encode_layout(
                &image,
                &options,
                0,
                &passes,
                responsive,
                false,
                1 << 20,
                None,
            )
            .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn streaming_area_global_rct_repeats_but_palettes_remain_local() {
        let image = f16_image(1025, 1);
        let mut options = streaming_options();
        options.rct = Some(6);
        for area in layout::frame_areas(1025, 1, 0).unwrap() {
            let prepared = prepare_frame_area(&image, &options, 0, area).unwrap();
            assert_eq!(prepared.global.transforms, [Transform::Rct(6)]);
            assert!(prepared.global.planes.is_empty());
            assert_complete(&prepared, area.width * 3);
            let (mut expected, _) = encode::planes(
                &image,
                Group {
                    x: area.x,
                    y: area.y,
                    width: area.width,
                    height: area.height,
                    stream_id: 0,
                },
                &raw_options(&options),
            )
            .unwrap();
            group::apply_rct(&mut expected, 0, 0, 6).unwrap();
            for (g, plan) in prepared.streams.iter().zip(&prepared.partition.streams) {
                assert_eq!(g.planes, layout::extract_channels(&expected, plan).unwrap());
                assert!(g.transforms.is_empty());
            }
        }
        let image = Image::new(
            1025,
            1,
            PixelLayout::Gray,
            PixelData::U8((0..1025).map(|i| if i % 2 == 0 { 7 } else { 91 }).collect()),
        )
        .unwrap();
        let mut options = streaming_options();
        let policy = options.palette_policy.as_mut().unwrap();
        policy.palette_colors = 8;
        policy.force_palette = true;
        policy.channel_colors_percent = 100.0;
        let area = layout::frame_areas(1025, 1, 0).unwrap()[0];
        let prepared = prepare_frame_area(&image, &options, 0, area).unwrap();
        assert!(prepared.global.transforms.is_empty());
        assert!(prepared.streams.iter().any(|g| {
            g.transforms
                .iter()
                .any(|t| matches!(t, Transform::Palette { .. }))
        }));
    }

    #[test]
    fn streaming_all_group_sizes_preserve_shifted_rows_across_dc_areas() {
        for shift in 0..=3 {
            let width = (1024usize << shift) + 1;
            let image = gray(width as u32, 9);
            let mut metadata = image.metadata().clone();
            let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
            extra.dim_shift = 3;
            metadata.extra_channels.push(extra);
            let extra_count = width.div_ceil(8) * 2;
            let image = image
                .with_metadata(
                    metadata,
                    vec![PixelData::U16(
                        (0..extra_count).map(|i| (i * 37) as u16).collect(),
                    )],
                )
                .unwrap();
            let options = streaming_options();
            let mut count = 0;
            for area in layout::frame_areas(width, 9, shift).unwrap() {
                let prepared = prepare_frame_area(&image, &options, shift, area).unwrap();
                for (g, plan) in prepared.streams.iter().zip(&prepared.partition.streams) {
                    for (plane, tile) in g.planes.iter().zip(&plan.channels) {
                        let channel_shift = image.channel_shift(tile.channel);
                        let (stride, _) = image.channel_dimensions(tile.channel);
                        let x0 = (area.x >> channel_shift) + tile.rect.x;
                        let y0 = (area.y >> channel_shift) + tile.rect.y;
                        for y in 0..plane.height {
                            for x in 0..plane.width {
                                assert_eq!(
                                    plane.samples[y * plane.width + x],
                                    image.modular_sample(tile.channel, (y0 + y) * stride + x0 + x)
                                );
                            }
                        }
                        count += plane.samples.len();
                    }
                }
            }
            assert_eq!(count, width * 9 + extra_count);
            let sections = encode_layout(
                &image,
                &options,
                shift,
                &Passes::default(),
                false,
                false,
                1 << 20,
                None,
            )
            .unwrap();
            assert_eq!(sections.len(), 13);
            assert!(sections[3].is_empty());
            assert!(
                sections
                    .iter()
                    .enumerate()
                    .all(|(i, s)| i == 3 || !s.is_empty())
            );
        }
    }

    #[test]
    fn streaming_dc_extra_and_tiny_last_area_encode_to_canonical_sections() {
        let image = gray(2049, 1);
        let mut metadata = image.metadata().clone();
        let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
        extra.dim_shift = 3;
        metadata.extra_channels.push(extra);
        let image = image
            .with_metadata(
                metadata,
                vec![PixelData::U16((0..257).map(|i| i * 37).collect())],
            )
            .unwrap();
        let options = streaming_options();
        let mut samples = 0;
        for area in layout::frame_areas(2049, 1, 1).unwrap() {
            let prepared = prepare_frame_area(&image, &options, 1, area).unwrap();
            assert!(prepared.global.planes.is_empty());
            assert_eq!(prepared.streams[0].planes.len(), 1);
            assert_eq!(prepared.streams[0].planes[0].hshift, 3);
            for (g, plan) in prepared.streams.iter().zip(&prepared.partition.streams) {
                for (channel, tile) in g.planes.iter().zip(&plan.channels) {
                    let offset = area.x >> image.channel_shift(tile.channel);
                    for (x, &value) in channel.samples.iter().enumerate() {
                        assert_eq!(
                            value,
                            image.modular_sample(tile.channel, offset + tile.rect.x + x)
                        );
                    }
                    samples += channel.samples.len();
                }
            }
        }
        assert_eq!(samples, 2049 + 257);
        let sections = encode_layout(
            &image,
            &options,
            1,
            &Passes::default(),
            false,
            false,
            1 << 20,
            None,
        )
        .unwrap();
        assert_eq!(sections.len(), 13);
        assert!(sections[3].is_empty()); // ACGlobal
        assert!(
            sections
                .iter()
                .enumerate()
                .all(|(i, s)| i == 3 || !s.is_empty())
        );
        assert_eq!(
            sections,
            encode_layout(
                &image,
                &options,
                1,
                &Passes::default(),
                false,
                true,
                1 << 20,
                None
            )
            .unwrap()
        );
        let mut source_tree = options.clone();
        source_tree.source_tree_entropy = true;
        let source_sections = encode_layout(
            &image,
            &source_tree,
            1,
            &Passes::default(),
            false,
            true,
            1 << 20,
            None,
        )
        .unwrap();
        assert_eq!(source_sections.len(), 13);
        assert!(
            source_sections
                .iter()
                .enumerate()
                .all(|(i, s)| i == 3 || !s.is_empty())
        );
        assert!(
            encode_layout(
                &image,
                &options,
                1,
                &Passes::default(),
                false,
                false,
                1,
                None
            )
            .is_err()
        );
        // Dynamic DC addressing fixes the source's literal /2048 assumption.
        let small_groups = encode_layout(
            &image,
            &options,
            0,
            &Passes::default(),
            false,
            false,
            1 << 20,
            None,
        )
        .unwrap();
        assert_eq!(small_groups.len(), 22);
        assert!(small_groups[4].is_empty());
        assert!(
            small_groups
                .iter()
                .enumerate()
                .all(|(i, s)| i == 4 || !s.is_empty())
        );
    }

    fn f16_image(width: u32, height: u32) -> Image {
        Image::new(
            width,
            height,
            PixelLayout::Rgb,
            PixelData::F16(
                (0..width as usize * height as usize * 3)
                    .map(|i| [0, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e23][i % 7])
                    .collect(),
            ),
        )
        .unwrap()
    }

    fn assert_complete(prepared: &PreparedLayout, count: usize) {
        let actual = std::iter::once(&prepared.global)
            .chain(&prepared.streams)
            .flat_map(|g| &g.planes)
            .map(|c| c.samples.len())
            .sum::<usize>();
        assert_eq!(actual, count);
        for (g, stream) in prepared.streams.iter().zip(&prepared.partition.streams) {
            assert_eq!(g.stream_id, stream.stream_id);
        }
    }

    #[test]
    fn source_integer_sign_bit_and_float_depth_bookkeeping() {
        for (pixels, bits, depth, max_value) in [
            (PixelData::U8(vec![0]), 8, 9, 255),
            (PixelData::U16(vec![0]), 16, 17, 65535),
            (PixelData::U32(vec![0]), 31, 32, i32::MAX),
            (PixelData::F16(vec![0]), 16, 16, 65535),
            (PixelData::F32(vec![-0.0]), 32, 32, 0),
        ] {
            let image = Image::new(1, 1, PixelLayout::Gray, pixels).unwrap();
            assert_eq!(image.bits(), bits);
            assert_eq!(
                effective_depth(&image).unwrap(),
                PaletteState {
                    meta_channels: 0,
                    max_bitdepth: depth,
                    max_value,
                }
            );
        }
    }

    #[test]
    fn f16_responsive_uses_global_rct_then_default_squeeze() {
        let image = f16_image(17, 9);
        for passes in [Passes::default(), Passes::progressive()] {
            let prepared = prepare_layout(&image, &options(), 0, &passes, true).unwrap();
            assert!(matches!(prepared.global.transforms.as_slice(),
                [Transform::Rct(6), Transform::SqueezeScript(p)] if p.is_empty()));
            assert_complete(&prepared, 17 * 9 * 3);
        }
        let mut metadata = image.metadata().clone();
        metadata.modular_16_bit_buffer_sufficient = true;
        let limited = image.with_metadata(metadata, Vec::new()).unwrap();
        let prepared = prepare_layout(&limited, &options(), 0, &Passes::default(), true).unwrap();
        assert!(prepared.global.transforms.is_empty());
    }

    #[test]
    fn f32_words_survive_palette_decline_and_insufficient_transform_headroom() {
        let words = [
            0, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc12345,
        ];
        let pixels = (0..257 * 17 * 3)
            .map(|i| f32::from_bits(words[i % words.len()]))
            .collect();
        let image = Image::new(257, 17, PixelLayout::Rgb, PixelData::F32(pixels)).unwrap();
        let mut options = options();
        options.palette_policy = Some(PaletteOptions::default());
        options.rct_candidates = Some(4);
        for responsive in [false, true] {
            let prepared =
                prepare_layout(&image, &options, 0, &Passes::progressive(), responsive).unwrap();
            assert!(prepared.global.planes.is_empty());
            assert!(prepared.global.transforms.is_empty());
            assert_complete(&prepared, 257 * 17 * 3);
            for (g, stream) in prepared.streams.iter().zip(&prepared.partition.streams) {
                assert!(g.transforms.is_empty());
                if !g.planes.is_empty() {
                    assert!(matches!(stream.kind, StreamKind::Ac { pass: 2, .. }));
                }
                for (channel, tile) in g.planes.iter().zip(&stream.channels) {
                    let image = &image;
                    let expected: Vec<_> = (tile.rect.y..tile.rect.y + tile.rect.height)
                        .flat_map(|y| {
                            (tile.rect.x..tile.rect.x + tile.rect.width)
                                .map(move |x| image.modular_sample(tile.channel, y * 257 + x))
                        })
                        .collect();
                    assert_eq!(channel.samples, expected);
                }
            }
        }
    }

    #[test]
    fn shifted_float_extra_raises_headroom_and_keeps_its_pass_geometry() {
        let image = f16_image(257, 17);
        let mut metadata = image.metadata().clone();
        let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
        extra.bit_depth = BitDepth::floating(32, 8).unwrap();
        extra.dim_shift = 1;
        metadata.extra_channels.push(extra);
        let image = image
            .with_metadata(metadata, vec![PixelData::F32(vec![-0.0; 129 * 9])])
            .unwrap();
        assert_eq!(
            effective_depth(&image).unwrap(),
            PaletteState {
                meta_channels: 0,
                max_bitdepth: 32,
                max_value: 65535,
            }
        );
        let prepared = prepare_layout(&image, &options(), 0, &Passes::progressive(), true).unwrap();
        assert!(prepared.global.transforms.is_empty());
        assert_complete(&prepared, 257 * 17 * 3 + 129 * 9);
        let mut extra_samples = 0;
        for (g, stream) in prepared.streams.iter().zip(&prepared.partition.streams) {
            for (channel, tile) in g.planes.iter().zip(&stream.channels) {
                if tile.channel == 3 {
                    assert!(matches!(stream.kind, StreamKind::Ac { pass: 1, .. }));
                    assert_eq!((channel.hshift, channel.vshift), (1, 1));
                    assert!(channel.samples.iter().all(|&v| v == i32::MIN));
                    extra_samples += channel.samples.len();
                }
            }
        }
        assert_eq!(extra_samples, 129 * 9);
    }

    #[test]
    fn integer_headroom_gates_rct_and_squeeze_without_rejecting_partition() {
        for (bits, rct, squeeze) in [
            (27, true, true),
            (28, true, false),
            (29, true, false),
            (30, false, false),
            (31, false, false),
        ] {
            let image =
                Image::new(17, 9, PixelLayout::Rgb, PixelData::U32(vec![7; 17 * 9 * 3])).unwrap();
            let mut metadata = image.metadata().clone();
            metadata.bit_depth = BitDepth::new(bits).unwrap();
            let image = image.with_metadata(metadata, Vec::new()).unwrap();
            let prepared = prepare_layout(&image, &options(), 0, &Passes::default(), true).unwrap();
            let mut expected = Vec::new();
            if rct {
                expected.push(Transform::Rct(6));
            }
            if squeeze {
                expected.push(Transform::SqueezeScript(Vec::new()));
            }
            assert_eq!(prepared.global.transforms, expected, "bits={bits}");
            assert_complete(&prepared, 17 * 9 * 3);
        }
    }

    #[test]
    fn responsive_default_produces_dc_and_each_ac_pass_without_losing_samples() {
        let pixels = (0..1033 * 65)
            .map(|i| (i * 17 + i / 1033 * 11) as u8)
            .collect();
        let image = Image::new(1033, 65, PixelLayout::Gray, PixelData::U8(pixels)).unwrap();
        let passes = Passes::progressive();
        let prepared = prepare_layout(&image, &options(), 0, &passes, true).unwrap();
        assert_complete(&prepared, 1033 * 65);
        assert!(
            matches!(prepared.global.transforms.as_slice(), [Transform::SqueezeScript(p)] if p.is_empty())
        );
        assert_eq!(prepared.partition.num_dc_groups, 2);
        assert_eq!(prepared.partition.num_groups, 9);
        assert!(
            prepared
                .streams
                .iter()
                .zip(&prepared.partition.streams)
                .any(|(g, s)| matches!(s.kind, StreamKind::Dc { .. }) && !g.planes.is_empty())
        );
        for pass in 0..3 {
            assert!(
                prepared
                    .streams
                    .iter()
                    .zip(&prepared.partition.streams)
                    .any(
                        |(g, s)| matches!(s.kind, StreamKind::Ac {pass: p, ..} if p == pass)
                            && !g.planes.is_empty()
                    )
            );
        }
    }

    #[test]
    fn sections_are_dc_then_acglobal_then_pass_major_not_preparation_order() {
        let passes = Passes::progressive();
        let shapes = [ChannelShape {
            width: 257,
            height: 17,
            hshift: 0,
            vshift: 0,
        }];
        let plan = layout::partition_channels(&shapes, 0, 257, 17, 0, &passes).unwrap();
        let sections = toc_sections(
            vec![255],
            plan.streams
                .iter()
                .map(|s| vec![s.stream_id as u8])
                .collect(),
            &plan,
            &passes,
            1024,
        )
        .unwrap();
        assert_eq!(plan.num_groups, 3);
        assert_eq!(sections.len(), 12);
        assert_eq!(sections[0], [255]);
        assert_eq!(sections[1], [2]);
        assert!(sections[2].is_empty());
        for pass in 0..3 {
            for group in 0..3 {
                assert_eq!(
                    sections[3 + pass * 3 + group],
                    [(21 + pass * 3 + group) as u8]
                );
            }
        }
        assert!(toc_sections(vec![255], Vec::new(), &plan, &passes, 1024).is_err());
        assert!(
            toc_sections(
                vec![255],
                vec![vec![0]; plan.streams.len()],
                &plan,
                &passes,
                1
            )
            .is_err()
        );
    }

    #[test]
    fn one_group_collapse_and_empty_local_streams_match_both_entropy_routes() {
        let image = f16_image(9, 9);
        let options = options();
        for shared in [false, true] {
            let one = encode_layout(
                &image,
                &options,
                0,
                &Passes::default(),
                true,
                shared,
                1 << 20,
                None,
            )
            .unwrap();
            assert_eq!(one.len(), 1);
            assert!(!one[0].is_empty());
            let three = encode_layout(
                &image,
                &options,
                0,
                &Passes::progressive(),
                true,
                shared,
                1 << 20,
                None,
            )
            .unwrap();
            assert_eq!(three.len(), 6);
            assert!(!three[0].is_empty());
            assert!(three[1..].iter().all(Vec::is_empty));
        }
    }
}
