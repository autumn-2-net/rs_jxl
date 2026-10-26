use crate::{
    Result, message, pixels,
    request::{Operation, Options, Request},
};
use rs_jxl::{
    EncodeOptions, Image,
    metadata::Animation,
    modular::predictor::Predictor,
    multiframe::{self, Reference},
    scheduler::{
        self, CandidateParams, NativeBackendOptions, adapter::project_native,
        native::encode_native_candidate, params::Override,
    },
};
use serde_json::{Value, json};

fn candidate(options: &Options) -> Result<(CandidateParams, EncodeOptions)> {
    let mut params = CandidateParams::default()
        .backend(options.effort.unwrap_or(7))
        .map_err(message)?;
    params.group_size_shift = options.group_size_shift;
    params.rct = options.rct;
    params.predictor = options
        .predictor
        .map(Predictor::try_from)
        .transpose()
        .map_err(message)?;
    if let Some(value) = options.palette_colors {
        params.palette_colors = value;
    }
    if let Some(value) = options.learning_fraction {
        params.learning_fraction = value;
    }
    if let Some(value) = options.patches {
        params.patches = if value { Override::On } else { Override::Off };
    }
    params.validate().map_err(message)?;
    let mut base = EncodeOptions {
        threads: options.threads.unwrap_or(1),
        ..Default::default()
    };
    base.modular.ma_max_leaves = options.ma_max_leaves;
    Ok((params, base))
}

fn still(image: &Image, options: &Options) -> Result<Vec<u8>> {
    let (params, base) = candidate(options)?;
    let settings = NativeBackendOptions::default();
    if options.source_search.unwrap_or(false) {
        if options.effort.unwrap_or(7) == 11 {
            Ok(scheduler::run_native_tectonic(
                image,
                &params,
                Default::default(),
                Default::default(),
                &base,
                settings,
            )
            .map_err(message)?
            .output
            .bytes)
        } else {
            Ok(scheduler::run_native_pixel_search(
                image,
                &params,
                &Default::default(),
                &base,
                settings,
            )
            .map_err(message)?
            .output
            .bytes)
        }
    } else {
        Ok(encode_native_candidate(&params, image, &base, settings)
            .map_err(message)?
            .bytes)
    }
}

fn reference(request: &Request) -> Reference {
    match request.reference.as_deref().unwrap_or("previous") {
        "full" => Reference::Full,
        "first" => Reference::First,
        "smallest_crop" => Reference::SmallestCrop,
        _ => Reference::Previous,
    }
}

fn rect(rect: Option<multiframe::Rect>) -> Value {
    rect.map_or(Value::Null, |r| {
        json!({
            "x": r.x,
            "y": r.y,
            "width": r.width,
            "height": r.height,
        })
    })
}

pub fn execute(request: &Request, data: &[u8]) -> Result<(Value, Vec<u8>)> {
    let mut images = Vec::new();
    let mut jpeg_input = false;
    for spec in request.images() {
        let (image, jpeg) =
            pixels::image(spec, data, request.options.strip_metadata.unwrap_or(false))?;
        images.push(image);
        jpeg_input |= jpeg;
    }
    if request.operation == Operation::Encode {
        let bytes = still(&images[0], &request.options)?;
        return Ok((
            json!({
                "operation": "encode",
                "bytes": bytes.len(),
                "jpeg_input": jpeg_input,
            }),
            bytes,
        ));
    }
    if request.operation == Operation::AnalyzeMultiframe {
        let mut frames = Vec::new();
        for (index, image) in images.iter().enumerate() {
            // The first frame is the full initial canvas, not an empty difference.
            let full = Some(multiframe::Rect {
                x: 0,
                y: 0,
                width: image.width(),
                height: image.height(),
            });
            let previous = if index == 0 {
                full
            } else {
                multiframe::difference_bounds(&images[index - 1], image).map_err(message)?
            };
            let first = if index == 0 {
                full
            } else {
                multiframe::difference_bounds(&images[0], image).map_err(message)?
            };
            frames.push(json!({
                "index": index,
                "previous": rect(previous),
                "first": rect(first),
            }));
        }
        return Ok((
            json!({
                "operation": "analyze_multiframe",
                "frames": frames,
            }),
            Vec::new(),
        ));
    }
    let timebase = request.timebase.unwrap_or([1000, 1]);
    let animation = Animation {
        ticks_per_second_numerator: timebase[0],
        ticks_per_second_denominator: timebase[1],
        loops: request.loops.unwrap_or(0),
        timecodes: false,
    };
    animation.validate().map_err(message)?;
    let images = images
        .into_iter()
        .map(|image| {
            let mut metadata = image.metadata().clone();
            metadata.animation = Some(animation);
            // with_metadata consumes the image but requires owned extra planes.
            let extra_planes = image.extra_planes().to_vec();
            image.with_metadata(metadata, extra_planes).map_err(message)
        })
        .collect::<Result<Vec<_>>>()?;
    let durations = request
        .durations
        .clone()
        .unwrap_or_else(|| vec![1; images.len()]);
    let (params, base) = candidate(&request.options)?;
    let options = project_native(&params, &images[0], &base)
        .map_err(message)?
        .exact_options()
        .map_err(message)?;
    let mut report = json!({
        "operation": "encode_multiframe",
        "jpeg_input": jpeg_input,
    });
    let bytes = if request.auto.unwrap_or(false) {
        let result = multiframe::encode_auto(
            &images,
            &durations,
            &options,
            &multiframe::AutoOptions {
                reference: reference(request),
                ..Default::default()
            },
        )
        .map_err(message)?;
        let frames: Vec<_> = result
            .frames
            .iter()
            .enumerate()
            .map(|(index, frame)| {
                let candidates: Vec<_> = frame
                    .candidates
                    .iter()
                    .map(|candidate| {
                        json!({
                            "strategy": format!("{:?}", candidate.strategy),
                            "encoded_bytes": candidate.encoded_bytes,
                            "encoded_frames": candidate.encoded_frames,
                        })
                    })
                    .collect();
                json!({
                    "index": index,
                    "winner_index": frame.winner_index,
                    "candidates": candidates,
                })
            })
            .collect();
        report["frames"] = json!(frames);
        result.bytes
    } else {
        let plan = multiframe::plan(&images, &durations, reference(request)).map_err(message)?;
        let frames: Vec<_> = plan.iter().map(|p| p.as_frame()).collect();
        rs_jxl::encode_frames(&images[0], &frames, &options).map_err(message)?
    };
    report["bytes"] = json!(bytes.len());
    Ok((report, bytes))
}

pub fn transcode(request: &Request, data: &[u8]) -> Result<(Value, Vec<u8>)> {
    use rs_jxl::jpeg::{self, cfl::CflMode, coefficient_order::OrderPolicy};
    let o = &request.options;
    let mut options = jpeg::TranscodeOptions {
        threads: o.threads.unwrap_or(1),
        effort: o.effort.unwrap_or(7),
        reconstruction_quality: o.brotli_quality,
        cfl: if o.no_cfl.unwrap_or(false) {
            CflMode::Disabled
        } else {
            CflMode::Enabled
        },
        order: match o.order.as_deref().unwrap_or("source") {
            "source" => OrderPolicy::Source,
            "default" => OrderPolicy::Default,
            "adaptive" => OrderPolicy::Adaptive,
            _ => return Err("order must be source/default/adaptive".into()),
        },
        ..Default::default()
    };
    if let Some(limits) = &o.limits {
        let target = &mut options.limits;
        if limits.max_metadata_bytes.is_some() {
            return Err("JPEG transcode does not expose a separate max_metadata_bytes limit; use max_input_bytes".into());
        }
        if let Some(n) = limits.max_input_bytes {
            target.input.structural.max_input_bytes = n;
        }
        if let Some(n) = limits.max_output_bytes {
            target.max_output_bytes = n;
        }
        if let Some(n) = limits.max_coefficients {
            target.max_coefficients = n;
        }
        if let Some(n) = limits.max_tokens {
            target.max_tokens = n;
        }
        if let Some(n) = limits.sample_limit {
            target.sample_limit = n;
        }
        if let Some(n) = limits.max_coefficient_bytes {
            target.input.max_coefficient_bytes = n;
        }
        if let Some(n) = limits.max_block_visits {
            target.input.max_block_visits = n;
        }
    }
    let bytes = if o.source_search.unwrap_or(false) {
        jpeg::transcode_search(data, &options, Default::default())
            .map_err(message)?
            .data
    } else {
        jpeg::transcode(data, &options).map_err(message)?
    };
    Ok((
        json!({
            "operation": "transcode_jpeg",
            "bytes": bytes.len(),
        }),
        bytes,
    ))
}
