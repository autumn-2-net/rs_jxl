use rs_jxl::decode::{
    DecodedBox, ExtraChannel, JxlBasicInfo, JxlBitDepth, JxlBlendInfo, JxlColorEncoding,
    JxlColorProfile, JxlExtraChannel, JxlFrameHeader, JxlPrimaries, JxlTransferFunction,
    JxlWhitePoint,
};
use serde_json::{Value, json};

pub(super) fn depth(depth: &JxlBitDepth) -> (u32, u32) {
    match depth {
        JxlBitDepth::Int { bits_per_sample } => (*bits_per_sample, 0),
        JxlBitDepth::Float {
            bits_per_sample,
            exponent_bits_per_sample,
        } => (*bits_per_sample, *exponent_bits_per_sample),
    }
}

pub(super) fn extra(info: &JxlExtraChannel, index: usize) -> Value {
    let kind = match info.ec_type {
        ExtraChannel::Alpha => "alpha",
        ExtraChannel::Depth => "depth",
        ExtraChannel::SpotColor => "spot_color",
        ExtraChannel::SelectionMask => "selection_mask",
        ExtraChannel::Black => "black",
        ExtraChannel::CFA => "cfa",
        ExtraChannel::Thermal => "thermal",
        ExtraChannel::Optional => "optional",
        _ => "unknown",
    };
    let (bits, exponent) = depth(&info.bit_depth);
    json!({
        "index": index,
        "type": kind,
        "type_code": info.ec_type as u32,
        "name": info.name,
        "bits_per_sample": bits,
        "exponent_bits_per_sample": exponent,
        "dim_shift": info.dim_shift,
        "alpha_premultiplied": info.alpha_associated,
        "spot_color": info.spot_color,
        "cfa_channel": info.cfa_channel,
    })
}

fn blend(info: &JxlBlendInfo) -> Value {
    json!({
        "mode": format!("{:?}", info.mode),
        "source": info.source,
        "alpha_channel": info.alpha_channel,
        "clamp": info.clamp,
    })
}

pub(super) fn header(frame: &JxlFrameHeader, index: usize) -> Value {
    let layer = &frame.layer_info;
    let extra_blends: Vec<_> = layer.extra_channel_blend_info.iter().map(blend).collect();
    json!({
        "index": index,
        "name": frame.name,
        "width": frame.size.0,
        "height": frame.size.1,
        "duration": frame.duration_ticks,
        "duration_ms": frame.duration,
        "is_last": frame.is_last,
        "is_preview": frame.is_preview,
        "timecode": frame.timecode,
        "layer_info": {
            "have_crop": layer.have_crop,
            "crop_origin": [layer.crop_origin.0, layer.crop_origin.1],
            "width": layer.size.0,
            "height": layer.size.1,
            "blend_info": blend(&layer.blend_info),
            "extra_channel_blend_info": extra_blends,
            "save_as_reference": layer.save_as_reference,
            "save_before_color_transform": layer.save_before_color_transform,
        },
    })
}

fn icc(profile: Option<&JxlColorProfile>) -> Value {
    profile
        .and_then(JxlColorProfile::try_as_icc)
        .map_or(Value::Null, |p| json!(p.as_slice()))
}

fn transfer(tf: &JxlTransferFunction) -> Value {
    match tf {
        JxlTransferFunction::BT709 => json!("bt709"),
        JxlTransferFunction::Linear => json!("linear"),
        JxlTransferFunction::SRGB => json!("srgb"),
        JxlTransferFunction::PQ => json!("pq"),
        JxlTransferFunction::DCI => json!("dci"),
        JxlTransferFunction::HLG => json!("hlg"),
        JxlTransferFunction::Gamma(g) => json!({"gamma":g}),
    }
}

fn color(profile: Option<&JxlColorProfile>) -> Value {
    match profile {
        None => Value::Null,
        Some(JxlColorProfile::Icc(_)) => json!({"kind":"icc"}),
        Some(JxlColorProfile::Simple(c)) => {
            let (space, wp, primaries, tf, intent) = match c {
                JxlColorEncoding::RgbColorSpace {
                    white_point,
                    primaries,
                    transfer_function,
                    rendering_intent,
                } => (
                    "rgb",
                    white_point,
                    Some(primaries),
                    transfer_function,
                    rendering_intent,
                ),
                JxlColorEncoding::GrayscaleColorSpace {
                    white_point,
                    transfer_function,
                    rendering_intent,
                } => (
                    "gray",
                    white_point,
                    None,
                    transfer_function,
                    rendering_intent,
                ),
                JxlColorEncoding::XYB { rendering_intent } => {
                    return json!({
                        "kind": "structured",
                        "color_space": "xyb",
                        "rendering_intent": format!("{:?}", rendering_intent),
                    });
                }
            };
            let standard =
                *wp == JxlWhitePoint::D65 && primaries.is_none_or(|p| *p == JxlPrimaries::SRGB);
            let name = if standard {
                match tf {
                    JxlTransferFunction::SRGB => Some("srgb"),
                    JxlTransferFunction::Linear => Some("linear_srgb"),
                    _ => None,
                }
            } else {
                None
            };
            json!({
                "kind": "structured",
                "name": name,
                "color_space": space,
                "white_point": wp.to_string(),
                "white_point_xy": wp.to_xy_coords(),
                "primaries": primaries.map(ToString::to_string),
                "primaries_xy": primaries.map(JxlPrimaries::to_xy_coords),
                "transfer_function": transfer(tf),
                "rendering_intent": format!("{:?}", intent),
            })
        }
    }
}

pub(super) fn input_color(color: &rs_jxl::color::ColorEncoding) -> Value {
    use rs_jxl::color::{ColorEncoding, TransferFunction};
    match color {
        ColorEncoding::Icc { .. } => json!({"kind":"icc"}),
        ColorEncoding::Fields(fields) => {
            let transfer_function = match fields.transfer_function {
                TransferFunction::Srgb => "srgb".to_owned(),
                TransferFunction::Linear => "linear".to_owned(),
                other => format!("{other:?}"),
            };
            json!({
                "kind": "structured",
                "color_space": format!("{:?}", fields.color_space).to_lowercase(),
                "white_point": format!("{:?}", fields.white_point),
                "primaries": format!("{:?}", fields.primaries),
                "transfer_function": transfer_function,
                "rendering_intent": format!("{:?}", fields.rendering_intent),
            })
        }
    }
}

pub(super) fn basic(
    info: &JxlBasicInfo,
    embedded: Option<&JxlColorProfile>,
    output: Option<&JxlColorProfile>,
    boxes: &[DecodedBox],
) -> Value {
    let (bits, exponent) = depth(&info.bit_depth);
    let extra_channels: Vec<_> = info
        .extra_channels
        .iter()
        .enumerate()
        .map(|(index, channel)| extra(channel, index))
        .collect();
    let animation = info.animation.as_ref().map(|animation| {
        json!({
            "tps_numerator": animation.tps_numerator,
            "tps_denominator": animation.tps_denominator,
            "num_loops": animation.num_loops,
            "have_timecodes": animation.have_timecodes,
        })
    });
    let boxes: Vec<_> = boxes
        .iter()
        .map(|metadata| {
            json!({
                "type": String::from_utf8_lossy(&metadata.kind),
                "data": metadata.data,
            })
        })
        .collect();
    json!({
        "width": info.size.0,
        "height": info.size.1,
        "bits_per_sample": bits,
        "exponent_bits_per_sample": exponent,
        "orientation": info.orientation as u32,
        "extra_channels": extra_channels,
        "animation": animation,
        "icc": icc(embedded),
        "output_icc": icc(output),
        "color_encoding": color(embedded),
        "output_color_encoding": color(output),
        "uses_original_profile": info.uses_original_profile,
        "intensity_target": info.tone_mapping.intensity_target,
        "min_nits": info.tone_mapping.min_nits,
        "relative_to_max_display": info.tone_mapping.relative_to_max_display,
        "linear_below": info.tone_mapping.linear_below,
        "boxes": boxes,
    })
}
