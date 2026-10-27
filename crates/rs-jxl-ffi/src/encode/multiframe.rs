use super::pixel_options;
use crate::{Result, message, request::Request};
use rs_jxl::{Image, codec::Encoder, metadata::Animation, multiframe};
use serde_json::{Value, json};

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

pub(super) fn auto_frame_report(index: usize, frame: &multiframe::AutoFrameReport) -> Value {
    let candidates: Vec<_> = frame
        .candidates
        .iter()
        .map(|candidate| {
            json!({
                "strategy": format!("{:?}", candidate.strategy),
                "tile_size": candidate.tile_size,
                "reference_index": candidate.reference_index,
                "source": candidate.source,
                "save_as_reference": candidate.save_as_reference,
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
                "tile_size": candidate.tile_size,
                "reference_index": candidate.reference_index,
                "source": candidate.source,
                "save_as_reference": candidate.save_as_reference,
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
                "source": difference.source,
                "changed_pixels": difference.changed_pixels,
                "bounds": rect(difference.bounds),
            })
        })
        .collect();
    json!({
        "index": index,
        "winner_index": frame.winner_index,
        "retained_reference": frame.retained_reference,
        "reference_lookahead": frame.reference_lookahead,
        "lookahead_comparisons": frame.lookahead_comparisons,
        "reused_payloads": frame.reused_payloads,
        "reused_payload_pixels": frame.reused_payload_pixels,
        "references_before": frame.references_before,
        "references_after": frame.references_after,
        "candidates": candidates,
        "pruned": pruned,
        "analysis_seconds": frame.analysis_elapsed.as_secs_f64(),
        "differences": differences,
    })
}

pub(super) fn execute(
    request: &Request,
    images: Vec<Image>,
    jpeg_input: bool,
    context: Option<&mut Encoder>,
) -> Result<(Value, Vec<u8>)> {
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
                reference: request.multiframe_reference(),
                ..Default::default()
            },
            |auto| auto.native(request.multiframe_reference()),
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
        let plan = multiframe::plan(&images, &durations, request.multiframe_reference())
            .map_err(message)?;
        let frames: Vec<_> = plan.iter().map(|p| p.as_frame()).collect();
        encoder
            .encode_frames(&images[0], &frames, &options)
            .map_err(message)?
    };
    report["bytes"] = json!(bytes.len());
    Ok((report, bytes))
}
