// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Whole-frame palette layout from enc_modular.cc: dictionary/meta channels
//! are coded in DCGlobal, index-plane tiles in independent Modular AC streams.
use super::{
    encode::{self, Group, ModularOptions},
    group::{self, Channel, Transform},
};
use crate::execution;
use crate::{Error, Image, Result, tools::patches::PatchDictionary};

type EncodedGroups = (Vec<u8>, Vec<Vec<u8>>);

/// Source global pre-transform palette -> explicit global RCT -> group split ->
/// local palette -> local RCT cost search. Global tables never become AC tiles.
pub(crate) fn encode_policy(
    image: &Image,
    groups: &[Group],
    options: &ModularOptions,
    limit: usize,
    patches: Option<&PatchDictionary>,
    shared: bool,
) -> Result<EncodedGroups> {
    use super::palette::{PaletteStage, PaletteState, apply_palettes};
    let policy = options
        .palette_policy
        .as_ref()
        .expect("source palette policy");
    if policy.responsive || policy.groupwise {
        return Err(Error::InvalidInput(
            "source progressive/groupwise transform orchestration is not integrated".into(),
        ));
    }
    let mut raw = options.clone();
    raw.rct = None;
    raw.rct_candidates = None;
    raw.palette_colors = None;
    raw.palette_predictor = None;
    raw.squeeze.clear();
    raw.palette_policy = None;
    let full = Group {
        x: 0,
        y: 0,
        width: image.width() as usize,
        height: image.height() as usize,
        stream_id: 0,
    };
    let (mut channels, _) = encode::planes(image, full, &raw)?;
    let mut state = PaletteState::new(image.bits())?;
    let mut transforms = apply_palettes(
        &mut channels,
        &mut state,
        image.bits(),
        PaletteStage::PreRct,
        policy,
    )?;
    if policy.do_color
        && channels.len() - state.meta_channels >= 3
        && options.rct_candidates.is_none()
        && let Some(kind) = options.rct
        && let Some(t) = group::apply_rct(
            &mut channels,
            state.meta_channels,
            state.meta_channels,
            kind,
        )?
    {
        transforms.push(t);
        state.max_bitdepth += 1;
    }
    let finish = |mut planes: Vec<Channel>,
                  mut transforms: Vec<Transform>,
                  mut state: PaletteState,
                  stream_id: usize,
                  local: bool|
     -> Result<encode::Prepared> {
        if local {
            transforms.extend(apply_palettes(
                &mut planes,
                &mut state,
                image.bits(),
                PaletteStage::PostRct,
                policy,
            )?);
        }
        if policy.do_color
            && planes.len() - state.meta_channels >= 3
            && let Some(count) = options.rct_candidates
        {
            let views: Vec<_> = planes
                .iter()
                .map(|p| super::cost::ChannelView {
                    pixels: &p.samples,
                    width: p.width,
                    height: p.height,
                    stride: p.width,
                })
                .collect();
            let color = &planes[state.meta_channels..state.meta_channels + 3];
            if color.iter().all(|p| {
                (p.width, p.height, p.hshift, p.vshift)
                    == (
                        color[0].width,
                        color[0].height,
                        color[0].hshift,
                        color[0].vshift,
                    )
            }) {
                let selected = super::cost::select_rct(&views, state.meta_channels, count)?;
                if let Some(t) = group::apply_rct(
                    &mut planes,
                    state.meta_channels,
                    state.meta_channels,
                    selected.rct_type,
                )? {
                    transforms.push(t);
                }
            }
        }
        for &horizontal in &options.squeeze {
            transforms.extend(group::apply_squeeze(
                &mut planes,
                state.meta_channels,
                horizontal,
            )?);
        }
        let wp = encode::weighted_header(&planes, options)?;
        Ok(encode::Prepared {
            planes,
            transforms,
            wp,
            stream_id,
        })
    };
    if groups.len() == 1 {
        let prepared = finish(channels, transforms, state, 0, false)?;
        if shared {
            return super::frame::encode_prepared_shared(
                None,
                vec![prepared],
                options,
                limit,
                patches,
                image.bits(),
            );
        }
        return Ok((
            vec![],
            vec![encode::encode_prepared(
                &prepared.planes,
                &prepared.transforms,
                options,
                limit,
                true,
                patches,
                0,
                image.bits(),
            )?],
        ));
    }
    let prepared = execution::try_map(groups, |_, g| {
        let mut planes = Vec::with_capacity(channels.len() - state.meta_channels);
        for full in &channels[state.meta_channels..] {
            if full.hshift != 0 || full.vshift != 0 {
                return Err(Error::InvalidInput(
                    "source palette partition expects full-resolution channels".into(),
                ));
            }
            let mut samples = Vec::with_capacity(g.width * g.height);
            for y in g.y..g.y + g.height {
                samples.extend_from_slice(
                    &full.samples[y * full.width + g.x..y * full.width + g.x + g.width],
                );
            }
            planes.push(Channel {
                width: g.width,
                height: g.height,
                hshift: 0,
                vshift: 0,
                samples,
            });
        }
        let mut local_state = state;
        local_state.meta_channels = 0;
        finish(planes, Vec::new(), local_state, g.stream_id, true)
    })?;
    channels.truncate(state.meta_channels);
    let global = encode::Prepared {
        planes: channels,
        transforms,
        wp: super::weighted::Header::default(),
        stream_id: 0,
    };
    if shared {
        return super::frame::encode_prepared_shared(
            Some(global),
            prepared,
            options,
            limit,
            patches,
            image.bits(),
        );
    }
    let mut global_options = options.clone();
    global_options.weighted_candidates = None;
    global_options.weighted_preset = 0;
    let global = encode::encode_prepared(
        &global.planes,
        &global.transforms,
        &global_options,
        limit,
        true,
        patches,
        0,
        image.bits(),
    )?;
    let encoded = execution::try_map(&prepared, |_, g| {
        encode::encode_prepared(
            &g.planes,
            &g.transforms,
            options,
            limit,
            false,
            None,
            g.stream_id,
            image.bits(),
        )
    })?;
    Ok((global, encoded))
}

pub(crate) fn encode_palette(
    image: &Image,
    groups: &[Group],
    options: &ModularOptions,
    limit: usize,
    patches: Option<&PatchDictionary>,
    shared: bool,
) -> Result<Option<EncodedGroups>> {
    if options.palette_colors.is_none() {
        return Ok(None);
    }
    let mut global_options = options.clone();
    global_options.rct = None;
    global_options.rct_candidates = None;
    global_options.squeeze.clear();
    let full = Group {
        x: 0,
        y: 0,
        width: image.width() as usize,
        height: image.height() as usize,
        stream_id: 0,
    };
    let (channels, transforms) = encode::planes(image, full, &global_options)?;
    if !transforms
        .iter()
        .any(|t| matches!(t, Transform::Palette { .. }))
    {
        return Ok(None);
    }
    if channels.len() != 2 {
        return Err(Error::Encoding("unexpected global palette layout".into()));
    }
    if groups.len() == 1 {
        if shared {
            let wp = encode::weighted_header(&channels, options)?;
            return super::frame::encode_prepared_shared(
                None,
                vec![encode::Prepared {
                    planes: channels,
                    transforms,
                    wp,
                    stream_id: 0,
                }],
                options,
                limit,
                patches,
                image.bits(),
            )
            .map(Some);
        }
        return Ok(Some((
            Vec::new(),
            vec![encode::encode_prepared(
                &channels,
                &transforms,
                options,
                limit,
                true,
                patches,
                0,
                image.bits(),
            )?],
        )));
    }
    let indices = &channels[1];
    let prepared = execution::try_map(groups, |_, g| {
        let mut samples = Vec::with_capacity(g.width * g.height);
        for y in g.y..g.y + g.height {
            samples.extend_from_slice(
                &indices.samples[y * indices.width + g.x..y * indices.width + g.x + g.width],
            );
        }
        let mut local = vec![Channel {
            width: g.width,
            height: g.height,
            hshift: 0,
            vshift: 0,
            samples,
        }];
        let mut transforms = Vec::new();
        for &horizontal in &options.squeeze {
            transforms.extend(group::apply_squeeze(&mut local, 0, horizontal)?);
        }
        let wp = encode::weighted_header(&local, options)?;
        Ok(encode::Prepared {
            planes: local,
            transforms,
            wp,
            stream_id: g.stream_id,
        })
    })?;
    if shared {
        let mut channels = channels;
        channels.truncate(1);
        let wp = encode::weighted_header(&channels, options)?;
        return super::frame::encode_prepared_shared(
            Some(encode::Prepared {
                planes: channels,
                transforms,
                wp,
                stream_id: 0,
            }),
            prepared,
            options,
            limit,
            patches,
            image.bits(),
        )
        .map(Some);
    }
    let global = encode::encode_prepared(
        &channels[..1],
        &transforms,
        options,
        limit,
        true,
        patches,
        0,
        image.bits(),
    )?;
    let encoded = execution::try_map(&prepared, |_, g| {
        encode::encode_prepared(
            &g.planes,
            &g.transforms,
            options,
            limit,
            false,
            None,
            g.stream_id,
            image.bits(),
        )
    })?;
    Ok(Some((global, encoded)))
}
