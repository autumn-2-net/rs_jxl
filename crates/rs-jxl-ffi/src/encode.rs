use crate::{
    Result, message, pixels,
    request::{Operation, Options, Request},
};
use rs_jxl::{
    EncodeOptions, Image,
    codec::{EncodeJob, EncodeReport, EncodeRoute, Encoder, SourceRequest, SourceSearch},
    modular::predictor::Predictor,
    scheduler::{
        CandidateParams, NativeBackendOptions, NativeImplementation, adapter::project_native,
        params::Override,
    },
};
use serde_json::{Value, json};

mod multiframe;

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

pub(crate) fn pixel_options(image: &Image, options: &Options) -> Result<EncodeOptions> {
    let (params, base) = candidate(options)?;
    project_native(&params, image, &base)
        .map_err(message)?
        .exact_options()
        .map_err(message)
}

fn still(
    image: &Image,
    options: &Options,
    context: Option<&mut Encoder>,
) -> Result<rs_jxl::codec::EncodeResult> {
    let (params, base) = candidate(options)?;
    let job = EncodeJob {
        options: base,
        source: Some(SourceRequest {
            params,
            search: if options.source_search.unwrap_or(false) {
                SourceSearch::Auto
            } else {
                SourceSearch::Disabled
            },
            backend: NativeBackendOptions::default(),
        }),
    };
    match context {
        Some(context) => context.encode(image, &job),
        None => rs_jxl::codec::encode_job(image, &job),
    }
    .map_err(message)
}

fn report(value: &EncodeReport) -> Value {
    let effective = value.effective_params.as_ref().map(|params| {
        json!({
            "effort": 10 - params.speed_tier,
            "group_size_shift": params.group_size_shift,
            "rct": params.rct,
            "predictor": params.predictor.map(|p| p as u32),
            "palette_colors": params.palette_colors,
            "learning_fraction": params.learning_fraction,
            "patches": match params.patches { Override::Default => "auto", Override::Off => "off", Override::On => "on" },
        })
    });
    json!({
        "route": match value.route {
            EncodeRoute::Manual => "manual", EncodeRoute::Candidate => "candidate",
            EncodeRoute::Search => "search", EncodeRoute::Scout => "scout",
            EncodeRoute::Pilot => "pilot", EncodeRoute::Tectonic => "tectonic",
        },
        "implementation": match value.implementation {
            NativeImplementation::Modular => "modular", NativeImplementation::FastLossless => "fast_lossless",
        },
        "pixels": value.pixels,
        "bytes": value.encoded_bytes,
        "bpp": value.bpp(),
        "codec_seconds": value.codec_seconds,
        "mpix_per_second": value.mpix_per_second(),
        "full_encodes": value.full_encodes,
        "probe_encodes": value.probe_encodes,
        "winner_index": value.winner_index,
        "effective_parameters": effective,
        "preparation_reuse": value.preparation_reuse.map(|r| json!({
            "hits": r.hits, "misses": r.misses, "retained_bytes": r.retained_bytes,
        })),
    })
}

pub fn execute(request: &Request, data: &[u8]) -> Result<(Value, Vec<u8>)> {
    execute_with_context(request, data, None)
}

pub fn execute_with_context(
    request: &Request,
    data: &[u8],
    context: Option<&mut Encoder>,
) -> Result<(Value, Vec<u8>)> {
    let mut images = Vec::new();
    let mut jpeg_input = false;
    for spec in request.images() {
        let (image, jpeg) =
            pixels::image(spec, data, request.options.strip_metadata.unwrap_or(false))?;
        images.push(image);
        jpeg_input |= jpeg;
    }
    if request.operation == Operation::Encode {
        let result = still(&images[0], &request.options, context)?;
        return Ok((
            json!({
                "operation": "encode",
                "bytes": result.bytes.len(),
                "jpeg_input": jpeg_input,
                "report": report(&result.report),
            }),
            result.bytes,
        ));
    }
    multiframe::execute(request, images, jpeg_input, context)
}

pub fn transcode(request: &Request, data: &[u8]) -> Result<(Value, Vec<u8>)> {
    transcode_with_context(request, data, None)
}

pub fn transcode_with_context(
    request: &Request,
    data: &[u8],
    context: Option<&mut Encoder>,
) -> Result<(Value, Vec<u8>)> {
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
    let search = o.source_search.unwrap_or(false).then(Default::default);
    let result = match context {
        Some(context) => context.transcode_jpeg(data, &options, search),
        None => rs_jxl::codec::transcode_jpeg(data, &options, search),
    }
    .map_err(message)?;
    Ok((
        json!({
            "operation": "transcode_jpeg",
            "bytes": result.bytes.len(),
            "report": {
                "route": if search.is_some() { "jpeg_search" } else { "jpeg_transcode" },
                "codec_seconds": result.codec_seconds,
                "backend_effort": result.backend_effort,
                "winner_index": result.winner_index,
                "full_encodes": result.trials.len(),
                "trials": result.trials.iter().map(|t| json!({
                    "stage": t.stage, "backend_effort": t.backend_effort, "bytes": t.bytes,
                })).collect::<Vec<_>>(),
            },
        }),
        result.bytes,
    ))
}

#[cfg(test)]
#[path = "encode/tests.rs"]
mod tests;
