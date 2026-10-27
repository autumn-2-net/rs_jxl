use crate::{
    Result, message, pixels,
    request::{Operation, Options, Request},
};
use rs_jxl::{
    EncodeOptions, Image,
    codec::{EncodeJob, EncodeReport, EncodeRoute, Encoder, SourceRequest, SourceSearch},
    metadata::Animation,
    modular::predictor::Predictor,
    multiframe::{self, Reference},
    scheduler::{
        CandidateParams, NativeBackendOptions, NativeImplementation, adapter::project_native,
        params::Override,
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

fn auto_frame_report(index: usize, frame: &multiframe::AutoFrameReport) -> Value {
    let candidates: Vec<_> = frame
        .candidates
        .iter()
        .map(|candidate| {
            json!({
                "strategy": format!("{:?}", candidate.strategy),
                "encoded_bytes": candidate.encoded_bytes,
                "encoded_frames": candidate.encoded_frames,
                "trial_pixels": candidate.trial_pixels,
                "elapsed_seconds": candidate.elapsed.as_secs_f64(),
            })
        })
        .collect();
    let pruned: Vec<_> = frame
        .pruned
        .iter()
        .map(|candidate| {
            json!({
                "strategy": format!("{:?}", candidate.strategy),
                "lower_bound": candidate.lower_bound,
                "encoded_frames": candidate.encoded_frames,
                "planned_frames": candidate.planned_frames,
                "trial_pixels": candidate.trial_pixels,
                "elapsed_seconds": candidate.elapsed.as_secs_f64(),
            })
        })
        .collect();
    let differences: Vec<_> = frame
        .differences
        .iter()
        .map(|difference| {
            json!({
                "reference": difference.reference,
                "changed_pixels": difference.changed_pixels,
                "bounds": rect(difference.bounds),
            })
        })
        .collect();
    json!({
        "index": index,
        "winner_index": frame.winner_index,
        "candidates": candidates,
        "pruned": pruned,
        "analysis_seconds": frame.analysis_elapsed.as_secs_f64(),
        "differences": differences,
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
    let options = pixel_options(&images[0], &request.options)?;
    let mut report = json!({
        "operation": "encode_multiframe",
        "jpeg_input": jpeg_input,
    });
    let mut owned;
    let encoder = match context {
        Some(encoder) => encoder,
        None => {
            owned = Encoder::new(options.threads).map_err(message)?;
            &mut owned
        }
    };
    let bytes = if request.auto.unwrap_or(false) {
        let auto = request.auto_options.as_ref().map_or_else(
            || multiframe::AutoOptions {
                reference: reference(request),
                ..Default::default()
            },
            |auto| auto.native(reference(request)),
        );
        let result = encoder
            .encode_auto(&images, &durations, &options, &auto)
            .map_err(message)?;
        let frames: Vec<_> = result
            .frames
            .iter()
            .enumerate()
            .map(|(index, frame)| auto_frame_report(index, frame))
            .collect();
        report["frames"] = json!(frames);
        result.bytes
    } else {
        let plan = multiframe::plan(&images, &durations, reference(request)).map_err(message)?;
        let frames: Vec<_> = plan.iter().map(|p| p.as_frame()).collect();
        encoder
            .encode_frames(&images[0], &frames, &options)
            .map_err(message)?
    };
    report["bytes"] = json!(bytes.len());
    Ok((report, bytes))
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
mod tests {
    use super::*;
    use multiframe::{
        AutoCandidateReport, AutoDiffReport, AutoFrameReport, AutoPrunedReport, Strategy,
    };
    use std::time::Duration;

    #[test]
    fn auto_report_keeps_native_statistics_and_separates_pruned_bounds() {
        let frame = AutoFrameReport {
            winner_index: 0,
            candidates: vec![AutoCandidateReport {
                strategy: Strategy::Full,
                encoded_bytes: 80,
                encoded_frames: 1,
                trial_pixels: 100,
                elapsed: Duration::from_millis(125),
            }],
            pruned: vec![AutoPrunedReport {
                strategy: Strategy::MergedFirst,
                lower_bound: 90,
                encoded_frames: 1,
                planned_frames: 3,
                trial_pixels: 30,
                elapsed: Duration::from_millis(250),
            }],
            analysis_elapsed: Duration::from_millis(500),
            differences: vec![
                AutoDiffReport {
                    reference: 0,
                    changed_pixels: 0,
                    bounds: None,
                },
                AutoDiffReport {
                    reference: 1,
                    changed_pixels: 2,
                    bounds: Some(multiframe::Rect {
                        x: 3,
                        y: 4,
                        width: 5,
                        height: 6,
                    }),
                },
            ],
        };
        assert_eq!(
            auto_frame_report(2, &frame),
            json!({
                "index": 2,
                "winner_index": 0,
                "candidates": [{
                    "strategy": "Full", "encoded_bytes": 80, "encoded_frames": 1,
                    "trial_pixels": 100, "elapsed_seconds": 0.125,
                }],
                "pruned": [{
                    "strategy": "MergedFirst", "lower_bound": 90,
                    "encoded_frames": 1, "planned_frames": 3,
                    "trial_pixels": 30, "elapsed_seconds": 0.25,
                }],
                "analysis_seconds": 0.5,
                "differences": [
                    {"reference": 0, "changed_pixels": 0, "bounds": null},
                    {"reference": 1, "changed_pixels": 2,
                     "bounds": {"x": 3, "y": 4, "width": 5, "height": 6}},
                ],
            })
        );
    }
}
