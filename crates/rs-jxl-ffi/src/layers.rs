use rs_jxl::{
    codec::Encoder,
    frame::{BlendInfo, BlendMode},
    layers::{self, LayerEncodeResult},
};
use serde_json::{Value, json};

use crate::{
    Result, encode, message, pixels,
    request::{LayerOptions, Request},
};

fn blend(value: &BlendInfo) -> Value {
    json!({
        "mode": match value.mode {
            BlendMode::Replace => "replace",
            BlendMode::Add => "add",
            BlendMode::Blend => "blend",
            BlendMode::AlphaWeightedAdd => "alpha_weighted_add",
            BlendMode::Multiply => "multiply",
        },
        "source": value.source,
        "alpha_channel": value.alpha_channel,
        "clamp": value.clamp,
    })
}

fn report(result: &LayerEncodeResult, jpeg_input: bool) -> Value {
    let layers: Vec<_> = result
        .layers
        .iter()
        .enumerate()
        .map(|(index, layer)| {
            let frame = &layer.options;
            json!({
                "index": index,
                "width": layer.width,
                "height": layer.height,
                "origin": [frame.origin.0, frame.origin.1],
                "duration": frame.duration,
                "timecode": frame.timecode,
                "name": frame.name,
                "blend": blend(&frame.blend),
                "extra_blend": frame.extra_blend.iter().map(blend).collect::<Vec<_>>(),
                "save_as_reference": frame.save_as_reference,
            })
        })
        .collect();
    json!({
        "operation": "encode_layers",
        "bytes": result.bytes.len(),
        "jpeg_input": jpeg_input,
        "canvas": {"width": result.width, "height": result.height},
        "codec_seconds": result.codec_seconds,
        "layers": layers,
    })
}

fn search_groups(request: &crate::request::Options) -> bool {
    request.effort.is_some_and(|effort| effort >= 9) && request.group_size_shift.is_none()
}

pub(crate) fn execute(
    request: &Request,
    data: &[u8],
    context: Option<&mut Encoder>,
) -> Result<(Value, Vec<u8>)> {
    let mut images = Vec::with_capacity(request.images().len());
    let mut jpeg_input = false;
    for spec in request.images() {
        let (image, jpeg) =
            pixels::image(spec, data, request.options.strip_metadata.unwrap_or(false))?;
        images.push(image);
        jpeg_input |= jpeg;
    }
    let first = images
        .first()
        .ok_or("encode_layers requires nonempty images")?;
    let options = encode::pixel_options(first, &request.options)?;
    let layers = request
        .layer_options
        .as_ref()
        .map_or_else(|| LayerOptions::default().native(), LayerOptions::native);
    let search_groups = search_groups(&request.options);
    let result = match context {
        Some(encoder) => {
            encoder.encode_layers_with_group_search(&images, &options, &layers, search_groups)
        }
        None => layers::encode_with_group_search(&images, &options, &layers, search_groups),
    }
    .map_err(message)?;
    Ok((report(&result, jpeg_input), result.bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_jxl::layers::{LayerInfo, default_frame_options};

    #[test]
    fn automatic_group_trials_keep_explicit_overrides() {
        let mut request = crate::request::Options {
            effort: Some(9),
            ..Default::default()
        };
        assert!(search_groups(&request));
        request.group_size_shift = Some(0);
        assert!(!search_groups(&request));
        request.group_size_shift = None;
        request.effort = Some(8);
        assert!(!search_groups(&request));
    }

    #[test]
    fn report_uses_native_fields_and_snake_case_blends() {
        for (mode, name) in [
            (BlendMode::Replace, "replace"),
            (BlendMode::Add, "add"),
            (BlendMode::Blend, "blend"),
            (BlendMode::AlphaWeightedAdd, "alpha_weighted_add"),
            (BlendMode::Multiply, "multiply"),
        ] {
            let mut options = default_frame_options();
            options.origin = (-2, 3);
            options.duration = 4;
            options.timecode = 5;
            options.name = "layer".into();
            options.blend = BlendInfo {
                mode,
                source: 2,
                alpha_channel: 1,
                clamp: true,
            };
            options.extra_blend = vec![options.blend];
            options.save_as_reference = 3;
            let result = LayerEncodeResult {
                bytes: vec![1, 2, 3],
                width: 10,
                height: 20,
                layers: vec![LayerInfo {
                    width: 6,
                    height: 7,
                    options,
                }],
                codec_seconds: 0.125,
            };
            let blend = json!({"mode": name, "source": 2, "alpha_channel": 1, "clamp": true});
            assert_eq!(
                report(&result, true),
                json!({
                    "operation": "encode_layers", "bytes": 3, "jpeg_input": true,
                    "canvas": {"width": 10, "height": 20}, "codec_seconds": 0.125,
                    "layers": [{
                        "index": 0, "width": 6, "height": 7, "origin": [-2, 3],
                        "duration": 4, "timecode": 5, "name": "layer",
                        "blend": blend, "extra_blend": [blend], "save_as_reference": 3,
                    }],
                })
            );
        }
    }
}
