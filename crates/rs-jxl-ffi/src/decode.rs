use crate::{
    Result, message, pixels,
    request::{Operation, Options, Request},
};
use rs_jxl::decode::{self, DecodeOptions, DitherMode, SampleFormat};
use serde_json::{Value, json};

mod metadata;
mod png;

pub(super) fn options(o: &Options) -> Result<DecodeOptions> {
    let mut options = DecodeOptions {
        threads: o.threads.unwrap_or(1),
        coalescing: o.coalescing.unwrap_or(true),
        adjust_orientation: o.adjust_orientation.unwrap_or(true),
        include_extra_channels: o.include_extra_channels.unwrap_or(true),
        // Returning native planes must not apply spot-color compositing to RGB.
        render_spot_colors: false,
        format: match o.format.as_deref().unwrap_or("native") {
            "native" => SampleFormat::Native,
            "u8" => SampleFormat::U8,
            "u16" => SampleFormat::U16,
            "f32" => SampleFormat::F32,
            _ => return Err("format must be native/u8/u16/f32".into()),
        },
        dither: match o.dither.as_deref().unwrap_or("jxl-rs") {
            "jxl-rs" => DitherMode::JxlRs,
            "libjxl" => DitherMode::Libjxl,
            _ => return Err("dither must be jxl-rs/libjxl".into()),
        },
        ..Default::default()
    };
    if let Some(limits) = &o.limits {
        macro_rules! apply { ($($field:ident),*) => { $(if let Some(value) = limits.$field { options.limits.$field = value; })* }; }
        apply!(
            max_input_bytes,
            max_output_bytes,
            max_metadata_bytes,
            max_pixels,
            max_samples,
            max_frames
        );
    }
    Ok(options)
}

pub fn execute(request: &Request, input: &[u8]) -> Result<(Value, Vec<u8>)> {
    if request.operation == Operation::ReadImage {
        return read_image(input, request.options.strip_metadata.unwrap_or(false));
    }
    let options = options(&request.options)?;
    if request.operation == Operation::ReconstructJpeg {
        let bytes = decode::reconstruct_jpeg(input, &options).map_err(message)?;
        return Ok((
            json!({
                "operation": "reconstruct_jpeg",
                "bytes": bytes.len(),
            }),
            bytes,
        ));
    }
    if request.operation == Operation::Info {
        let image = decode::inspect(input, &options).map_err(message)?;
        let mut result = metadata::basic(
            &image.info,
            image.embedded_profile.as_ref(),
            image.output_profile.as_ref(),
            &image.boxes,
        );
        result["operation"] = json!("info");
        result["frames"] = json!(
            image
                .frames
                .iter()
                .enumerate()
                .map(|(i, f)| metadata::header(f, i))
                .collect::<Vec<_>>()
        );
        result["preview"] = image
            .preview
            .as_ref()
            .map_or(Value::Null, |f| metadata::header(f, 0));
        result["bytes_read"] = json!(image.bytes_read);
        result["complete"] = json!(true);
        return Ok((result, Vec::new()));
    }
    let target = if request.operation == Operation::DecodeToPng {
        Some(request.options.frame_index.unwrap_or(0))
    } else {
        request.options.frame_index
    };
    let image = match target {
        Some(index) => decode::decode_frame(input, index, &options),
        None => decode::decode(input, &options),
    }
    .map_err(message)?;
    if request.operation == Operation::DecodeToPng {
        let bytes = png::encode(&image, &options)?;
        return Ok((
            json!({
                "operation": "decode_to_png",
                "bytes": bytes.len(),
                "frame_index": target,
            }),
            bytes,
        ));
    }
    let mut result = metadata::basic(
        &image.info,
        image.embedded_profile.as_ref(),
        image.output_profile.as_ref(),
        &image.boxes,
    );
    result["operation"] = json!("decode");
    result["complete"] = json!(image.complete);
    result["bytes_read"] = json!(image.bytes_read);
    result["output_intensity_target"] = json!(image.output_intensity_target);
    let mut data = Vec::new();
    let mut frames = Vec::new();
    for frame in &image.frames {
        let mut value = pixels::append(
            &mut data,
            &frame.pixels,
            frame.width,
            frame.height,
            frame.layout.channels(),
        )?;
        value["index"] = json!(frame.index);
        value["duration"] = json!(frame.duration_ticks);
        value["name"] = json!(frame.name);
        value["complete"] = json!(frame.complete);
        value["header"] = metadata::header(&frame.header, frame.index);
        let mut extras = Vec::new();
        if options.include_extra_channels
            && frame.extra_planes.len() != image.info.extra_channels.len()
        {
            return Err("decoder extra planes do not match metadata".into());
        }
        for (index, (plane, info)) in frame
            .extra_planes
            .iter()
            .zip(&image.info.extra_channels)
            .enumerate()
        {
            let mut extra = pixels::append(&mut data, plane, frame.width, frame.height, 1)?;
            pixels::merge(&mut extra, metadata::extra(info, index));
            extras.push(extra);
        }
        value["extra_channels"] = json!(extras);
        frames.push(value);
    }
    if data.len() > options.limits.max_output_bytes {
        return Err("packed output exceeds max_output_bytes".into());
    }
    result["frames"] = json!(frames);
    Ok((result, data))
}

fn read_image(input: &[u8], strip: bool) -> Result<(Value, Vec<u8>)> {
    let decoded = rs_jxl_image::decode(input, strip).map_err(message)?;
    let image = decoded.image;
    let meta = image.metadata();
    let mut data = Vec::new();
    let mut frame = pixels::append(
        &mut data,
        image.pixels(),
        image.width() as usize,
        image.height() as usize,
        image.layout().channels(),
    )?;
    let mut extras = Vec::new();
    let alpha = usize::from(image.layout().has_alpha());
    for (index, info) in meta.extra_channels.iter().enumerate() {
        let plane;
        let pixels = if index < alpha {
            plane = pixels::alpha_plane(image.pixels(), image.layout().channels());
            &plane
        } else {
            &image.extra_planes()[index - alpha]
        };
        let (width, height) = image.channel_dimensions(image.color_channels() + index);
        let mut value = pixels::append(&mut data, pixels, width, height, 1)?;
        pixels::merge(&mut value, pixels::extra_metadata(info, index));
        extras.push(value);
    }
    frame["index"] = json!(0);
    frame["name"] = json!("");
    frame["duration"] = json!(0);
    frame["complete"] = json!(true);
    frame["extra_channels"] = json!(extras);
    frame["header"] = json!({
        "index": 0,
        "name": "",
        "width": image.width(),
        "height": image.height(),
        "duration": 0,
        "is_last": true,
        "is_preview": false,
        "timecode": null,
        "layer_info": {
            "have_crop": false,
            "crop_origin": [0, 0],
            "width": image.width(),
            "height": image.height(),
            "blend_info": {
                "mode": "Replace",
                "source": 0,
                "alpha_channel": 0,
                "clamp": false,
            },
            "extra_channel_blend_info": [],
            "save_as_reference": null,
            "save_before_color_transform": false,
        },
    });
    let color = metadata::input_color(&meta.color_encoding);
    let extra_channels: Vec<_> = meta
        .extra_channels
        .iter()
        .enumerate()
        .map(|(index, info)| pixels::extra_metadata(info, index))
        .collect();
    Ok((
        json!({
            "operation": "read_image",
            "width": image.width(),
            "height": image.height(),
            "channels": image.layout().channels(),
            "bits_per_sample": meta.bit_depth.bits_per_sample,
            "exponent_bits_per_sample": meta.bit_depth.exponent_bits_per_sample,
            "orientation": meta.orientation as u8,
            "extra_channels": extra_channels,
            "animation": null,
            "icc": meta.color_encoding.icc_profile(),
            "output_icc": meta.color_encoding.icc_profile(),
            "color_encoding": color,
            "output_color_encoding": color,
            "boxes": [],
            "bytes_read": input.len(),
            "complete": true,
            "jpeg_input": decoded.jpeg,
            "frames": [frame],
        }),
        data,
    ))
}
