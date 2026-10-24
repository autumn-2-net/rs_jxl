//! Metadata extraction and TIFF orientation handling.
use super::writers::Output;
use super::*;
use rs_jxl::decode::{ImageInspection, JxlColorProfile, JxlFrameHeader};
use serde_json::{Value, json};
use std::io::Write;

pub(super) struct Metadata<'a> {
    pub info: &'a JxlBasicInfo,
    pub frames: Vec<&'a JxlFrameHeader>,
    pub embedded_profile: Option<&'a JxlColorProfile>,
    pub output_profile: Option<&'a JxlColorProfile>,
    pub output_intensity_target: f32,
    pub boxes: &'a [DecodedBox],
    pub bytes_read: usize,
    pub complete: bool,
}

impl<'a> From<&'a ImageInspection> for Metadata<'a> {
    fn from(image: &'a ImageInspection) -> Self {
        Self {
            info: &image.info,
            frames: image.frames.iter().collect(),
            embedded_profile: image.embedded_profile.as_ref(),
            output_profile: image.output_profile.as_ref(),
            output_intensity_target: image.info.tone_mapping.intensity_target,
            boxes: &image.boxes,
            bytes_read: image.bytes_read,
            complete: true,
        }
    }
}

impl<'a> From<&'a DecodedImage> for Metadata<'a> {
    fn from(image: &'a DecodedImage) -> Self {
        Self {
            info: &image.info,
            frames: image.frames.iter().map(|f| &f.header).collect(),
            embedded_profile: image.embedded_profile.as_ref(),
            output_profile: image.output_profile.as_ref(),
            output_intensity_target: image.output_intensity_target,
            boxes: &image.boxes,
            bytes_read: image.bytes_read,
            complete: image.complete,
        }
    }
}

pub(super) fn extract(boxes: &[DecodedBox], format: OutputFormat, limit: usize) -> Result<Vec<u8>> {
    let kind = match format {
        OutputFormat::Exif => b"Exif",
        OutputFormat::Xmp => b"xml ",
        OutputFormat::Jumbf => b"jumb",
        _ => return Err("not a metadata format".into()),
    };
    let mut matching = boxes.iter().filter(|b| &b.kind == kind);
    let data = &matching.next().ok_or("requested metadata is absent")?.data;
    if matching.next().is_some() {
        return Err("multiple matching metadata boxes cannot be represented by one sidecar without ambiguity".into());
    }
    let data = if format == OutputFormat::Exif {
        let offset = u32::from_be_bytes(data.get(..4).ok_or("truncated EXIF offset")?.try_into()?);
        let start = usize::try_from(offset)?
            .checked_add(4)
            .ok_or("EXIF offset overflow")?;
        let tiff = data.get(start..).ok_or("EXIF offset exceeds box length")?;
        if !matches!(tiff.get(..4), Some(b"II\x2a\0" | b"MM\0\x2a")) {
            return Err("EXIF lacks a TIFF header".into());
        }
        tiff
    } else {
        data.as_slice()
    };
    let mut output = Output::with_limit(limit);
    output.write_all(data)?;
    Ok(output.into_bytes())
}

pub(super) fn icc(profile: Option<&JxlColorProfile>, limit: usize) -> Result<Vec<u8>> {
    let profile = profile.ok_or("requested color profile is unavailable")?;
    let icc = profile
        .try_as_icc()
        .ok_or("requested profile cannot be represented as ICC")?;
    rs_jxl::icc::validate_profile(icc.as_slice())?;
    let mut output = Output::with_limit(limit);
    output.write_all(icc.as_slice())?;
    Ok(output.into_bytes())
}

fn depth(depth: &JxlBitDepth) -> (u32, u32) {
    match depth {
        JxlBitDepth::Int { bits_per_sample } => (*bits_per_sample, 0),
        JxlBitDepth::Float {
            bits_per_sample,
            exponent_bits_per_sample,
        } => (*bits_per_sample, *exponent_bits_per_sample),
    }
}

pub(super) fn json_metadata(metadata: &Metadata<'_>, limit: usize) -> Result<Vec<u8>> {
    let info = metadata.info;
    let depths: Vec<_> = std::iter::once(&info.bit_depth)
        .chain(info.extra_channels.iter().map(|ec| &ec.bit_depth))
        .map(depth)
        .collect();
    let frames: Vec<Value> = metadata.frames.iter().enumerate().map(|(index, frame)| {
        let layer = &frame.layer_info;
        let blend = |blend: &rs_jxl::decode::JxlBlendInfo| json!({"mode": format!("{:?}", blend.mode), "source": blend.source,
            "alpha_channel": blend.alpha_channel, "clamp": blend.clamp});
        let mut value = json!({"index": index, "name": frame.name, "width": frame.size.0, "height": frame.size.1,
            "duration_ticks": frame.duration_ticks, "is_last": frame.is_last, "is_preview": frame.is_preview,
            "layer_info": {"have_crop": layer.have_crop, "crop_origin": [layer.crop_origin.0, layer.crop_origin.1],
                "width": layer.size.0, "height": layer.size.1, "blend_info": blend(&layer.blend_info),
                "extra_channel_blend_info": layer.extra_channel_blend_info.iter().map(blend).collect::<Vec<_>>(),
                "save_as_reference": layer.save_as_reference, "save_before_color_transform": layer.save_before_color_transform}});
        if let Some(duration) = frame.duration { value["duration"] = json!(duration / 1000.0); }
        if let Some(timecode) = frame.timecode { value["timecode"] = json!(timecode); }
        value
    }).collect();
    let extras: Vec<_> = info.extra_channels.iter().enumerate().map(|(index, ec)| json!({
        "index": index, "name": ec.name, "type": format!("{:?}", ec.ec_type),
        "bits_per_sample": depth(&ec.bit_depth).0, "exp_bits_per_sample": depth(&ec.bit_depth).1,
        "dim_shift": ec.dim_shift, "alpha_associated": ec.alpha_associated,
        "spot_color": ec.spot_color, "cfa_channel": ec.cfa_channel,
    })).collect();
    let boxes: Vec<_> = metadata
        .boxes
        .iter()
        .map(|b| json!({"type": String::from_utf8_lossy(&b.kind), "size": b.data.len()}))
        .collect();
    let mut value = json!({
        "width": info.size.0, "height": info.size.1, "orientation": info.orientation as u32,
        "frames": frames, "complete": metadata.complete, "bytes_read": metadata.bytes_read,
        "intensity_target": info.tone_mapping.intensity_target, "min_nits": info.tone_mapping.min_nits,
        "output_intensity_target": metadata.output_intensity_target,
        "relative_to_max_display": info.tone_mapping.relative_to_max_display, "linear_below": info.tone_mapping.linear_below,
        "uses_original_profile": info.uses_original_profile,
        "bits_per_sample": depths.iter().map(|d| d.0).collect::<Vec<_>>(),
        "exp_bits_per_sample": depths.iter().map(|d| d.1).collect::<Vec<_>>(),
        "extra_channel_type": info.extra_channels.iter().map(|ec| format!("{:?}", ec.ec_type)).collect::<Vec<_>>(),
        "extra_channels": extras, "boxes": boxes,
    });
    if let Some(size) = info.preview_size {
        value["preview"] = json!({"width": size.0, "height": size.1});
    }
    if let Some(animation) = &info.animation {
        value["animation"] = json!({"tps_numerator": animation.tps_numerator, "tps_denominator": animation.tps_denominator,
            "num_loops": animation.num_loops, "have_timecodes": animation.have_timecodes});
    }
    let mut output = Output::with_limit(limit);
    serde_json::to_writer_pretty(&mut output, &value)?;
    output.write_all(b"\n")?;
    Ok(output.into_bytes())
}

pub(in crate::decode) fn orientation_exif(orientation: u16) -> Vec<u8> {
    let mut tiff = b"II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x01\0\0\0\0\0\0\0".to_vec();
    tiff[18..20].copy_from_slice(&orientation.to_le_bytes());
    tiff
}

pub(in crate::decode) fn png_exif(data: &[u8], orientation: u16) -> Result<Vec<u8>> {
    let offset = u32::from_be_bytes(
        data.get(..4)
            .ok_or("truncated JXL EXIF offset")?
            .try_into()?,
    );
    let start = usize::try_from(offset)?
        .checked_add(4)
        .ok_or("EXIF offset overflow")?;
    let mut tiff = data
        .get(start..)
        .ok_or("JXL EXIF offset is outside the box")?
        .to_vec();
    let little = match tiff.get(..4) {
        Some(b"II\x2a\0") => true,
        Some(b"MM\0\x2a") => false,
        _ => return Err("EXIF does not contain a supported TIFF header".into()),
    };
    let word = |bytes: &[u8]| -> u16 {
        let bytes = [bytes[0], bytes[1]];
        if little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        }
    };
    let dword = |bytes: &[u8]| -> u32 {
        let bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        }
    };
    let encode_word = |value: u16| {
        if little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        }
    };
    let encode_dword = |value: u32| {
        if little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        }
    };
    let directory = usize::try_from(dword(tiff.get(4..8).ok_or("truncated EXIF header")?))?;
    if directory < 8 {
        return Err("invalid EXIF IFD0 offset".into());
    }
    let count = word(
        tiff.get(directory..directory.checked_add(2).ok_or("EXIF offset overflow")?)
            .ok_or("truncated EXIF IFD0")?,
    );
    let entries = directory.checked_add(2).ok_or("EXIF offset overflow")?;
    let end = entries
        .checked_add(usize::from(count) * 12)
        .ok_or("EXIF directory overflow")?;
    let next_end = end
        .checked_add(4)
        .filter(|&n| n <= tiff.len())
        .ok_or("truncated EXIF directory")?;
    let mut found = false;
    for index in 0..usize::from(count) {
        let entry = entries + index * 12;
        if word(&tiff[entry..entry + 2]) == 274 {
            if word(&tiff[entry + 2..entry + 4]) != 3 || dword(&tiff[entry + 4..entry + 8]) != 1 {
                return Err("malformed EXIF orientation tag".into());
            }
            tiff[entry + 8..entry + 10].copy_from_slice(&encode_word(orientation));
            found = true;
        }
    }
    if !found && orientation != 1 {
        // Append a replacement IFD0 so every existing absolute TIFF offset remains valid.
        let new_count = count.checked_add(1).ok_or("EXIF IFD0 is full")?;
        if tiff.len() % 2 != 0 {
            tiff.push(0);
        }
        let new_offset = u32::try_from(tiff.len())?;
        let mut replacement = Vec::new();
        replacement.extend_from_slice(&encode_word(new_count));
        let insertion = (0..usize::from(count))
            .find(|&index| word(&tiff[entries + index * 12..entries + index * 12 + 2]) > 274)
            .unwrap_or(usize::from(count));
        let split = entries + insertion * 12;
        replacement.extend_from_slice(&tiff[entries..split]);
        replacement.extend_from_slice(&encode_word(274));
        replacement.extend_from_slice(&encode_word(3));
        replacement.extend_from_slice(&encode_dword(1));
        replacement.extend_from_slice(&encode_word(orientation));
        replacement.extend_from_slice(&[0, 0]);
        replacement.extend_from_slice(&tiff[split..next_end]);
        tiff.extend_from_slice(&replacement);
        tiff[4..8].copy_from_slice(&encode_dword(new_offset));
    }
    Ok(tiff)
}
