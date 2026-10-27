//! Text and JSON rendering of automatic multiframe candidate reports.

use serde_json::{Value, json};
use std::io::Write;

use rs_jxl::multiframe::AutoEncodeResult;

use crate::Result;

pub(super) fn write_report(
    mut output: impl Write,
    result: &AutoEncodeResult,
) -> std::io::Result<()> {
    for (frame_index, frame) in result.frames.iter().enumerate() {
        writeln!(
            output,
            "frame {}: winner={} candidates={} pruned={} analysis_elapsed={:?} retained_reference={} reference_lookahead={} lookahead_comparisons={} reused_payloads={} reused_payload_pixels={} references_before={:?} references_after={:?}",
            frame_index + 1,
            frame.winner_index,
            frame.candidates.len(),
            frame.pruned.len(),
            frame.analysis_elapsed,
            frame.retained_reference,
            frame.reference_lookahead,
            frame.lookahead_comparisons,
            frame.reused_payloads,
            frame.reused_payload_pixels,
            frame.references_before,
            frame.references_after,
        )?;
        for difference in &frame.differences {
            writeln!(
                output,
                "  difference reference={} source={} changed_pixels={} bounds={:?}",
                difference.reference,
                difference.source,
                difference.changed_pixels,
                difference.bounds,
            )?;
        }
        for (index, candidate) in frame.candidates.iter().enumerate() {
            writeln!(
                output,
                "  candidate={index} strategy={:?} reference_index={:?} source={} save_as_reference={} encoded_bytes={} encoded_frames={} trial_pixels={} elapsed={:?}{}",
                candidate.strategy,
                candidate.reference_index,
                candidate.source,
                candidate.save_as_reference,
                candidate.encoded_bytes,
                candidate.encoded_frames,
                candidate.trial_pixels,
                candidate.elapsed,
                if index == frame.winner_index {
                    " winner"
                } else {
                    ""
                }
            )?;
        }
        for (index, candidate) in frame.pruned.iter().enumerate() {
            writeln!(
                output,
                "  pruned={index} strategy={:?} reference_index={:?} source={} save_as_reference={} lower_bound={} encoded_frames={} planned_frames={} trial_pixels={} elapsed={:?}",
                candidate.strategy,
                candidate.reference_index,
                candidate.source,
                candidate.save_as_reference,
                candidate.lower_bound,
                candidate.encoded_frames,
                candidate.planned_frames,
                candidate.trial_pixels,
                candidate.elapsed,
            )?;
        }
    }
    writeln!(
        output,
        "sequence: inputs={} output_bytes={}",
        result.frames.len(),
        result.bytes.len()
    )
}

fn candidate_report(candidate: &rs_jxl::multiframe::AutoCandidateReport) -> Value {
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
}

fn pruned_report(candidate: &rs_jxl::multiframe::AutoPrunedReport) -> Value {
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
}

fn difference_report(diff: &rs_jxl::multiframe::AutoDiffReport) -> Value {
    json!({
        "reference": diff.reference,
        "source": diff.source,
        "changed_pixels": diff.changed_pixels,
        "bounds": diff.bounds.map(|rect| json!({
            "x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height,
        })),
    })
}

fn frame_report(index: usize, frame: &rs_jxl::multiframe::AutoFrameReport) -> Value {
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
        "analysis_elapsed_seconds": frame.analysis_elapsed.as_secs_f64(),
        "candidates": frame.candidates.iter().map(candidate_report).collect::<Vec<_>>(),
        "pruned": frame.pruned.iter().map(pruned_report).collect::<Vec<_>>(),
        "differences": frame.differences.iter().map(difference_report).collect::<Vec<_>>(),
    })
}

pub(super) fn write_json_report(mut output: impl Write, result: &AutoEncodeResult) -> Result<()> {
    let frames: Vec<_> = result
        .frames
        .iter()
        .enumerate()
        .map(|(index, frame)| frame_report(index, frame))
        .collect();
    serde_json::to_writer(
        &mut output,
        &json!({
            "operation": "encode_frames",
            "inputs": result.frames.len(),
            "output_bytes": result.bytes.len(),
            "frames": frames,
        }),
    )?;
    writeln!(output)?;
    Ok(())
}
