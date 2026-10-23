//! Original-byte reconstruction from a complete JXL container, never pixels.
use super::{DecodeOptions, DecodedBox, backend_options, boxes, failure, runner};
use crate::jpeg::reconstruction_decode::{
    CapturedJpegCoefficients, CapturedJpegComponent, ReconstructionDecodeLimits,
};
use crate::jpeg::reconstruction_metadata::ReconstructionMetadataSources;
use crate::{Error, Result};
use jxl_decoder::api::{Event, JxlColorProfile, JxlDecoder, JxlJpegCoefficientCapture};

fn unique_box(boxes: &[DecodedBox], kind: [u8; 4]) -> Result<Option<&[u8]>> {
    let mut found = boxes.iter().filter(|entry| entry.kind == kind);
    let first = found.next().map(|entry| entry.data.as_slice());
    if found.next().is_some() {
        return Err(failure(
            "ambiguous duplicate JPEG reconstruction source box",
        ));
    }
    Ok(first)
}

/// Reconstruct the original JPEG byte stream from JXL coefficients and `jbrd`.
/// Requires a supported JPEG-transcode frame; pixel-only JXL is an error. No
/// lossy JPEG re-encoding fallback and no access to the original input JPEG.
/// Sample output format and orientation settings never alter JPEG bytes.
pub fn reconstruct_jpeg(mut input: &[u8], options: &DecodeOptions) -> Result<Vec<u8>> {
    let mut backend = backend_options(input, options)?;
    backend.adjust_orientation = false;
    backend.render_spot_colors = false;
    backend.request_aux_boxes = boxes::REQUESTED.to_vec();
    let mut decoder = JxlDecoder::new(backend);
    decoder.close_input();
    let mut pool = runner::Runner::new(options.threads)?;
    let mut capture: Option<JxlJpegCoefficientCapture> = None;
    let mut frame_complete = false;
    loop {
        match decoder
            .process(&mut input, None, Some(&mut pool))
            .map_err(failure)?
        {
            Event::BasicInfo => {
                let info = decoder
                    .basic_info()
                    .ok_or_else(|| failure("missing basic info"))?;
                let (width, height) = info.size;
                let pixels = width
                    .checked_mul(height)
                    .ok_or(Error::Limit("JPEG pixel count overflow"))?;
                if pixels > options.limits.max_pixels {
                    return Err(Error::Limit("JPEG pixel count"));
                }
                // Conservatively include MCU padding for all supported sampling factors.
                let padded = width
                    .div_ceil(32)
                    .checked_mul(32)
                    .and_then(|w| {
                        height
                            .div_ceil(32)
                            .checked_mul(32)
                            .and_then(|h| w.checked_mul(h))
                    })
                    .and_then(|n| n.checked_mul(3))
                    .ok_or(Error::Limit("JPEG coefficient allocation overflow"))?;
                if padded > options.limits.max_samples
                    || padded > options.limits.max_output_bytes / 2
                {
                    return Err(Error::Limit("JPEG coefficient allocation"));
                }
            }
            Event::FrameHeader => {
                if capture.is_some() || frame_complete {
                    return Err(failure("JPEG reconstruction requires one frame"));
                }
                capture = Some(decoder.enable_jpeg_coefficient_capture().map_err(failure)?);
            }
            Event::FrameComplete { has_more_frames } => {
                if has_more_frames {
                    return Err(failure("JPEG reconstruction does not accept animations"));
                }
                frame_complete = true;
            }
            Event::Complete => break,
            Event::NeedMoreInput { .. } => {
                return Err(failure("truncated JPEG reconstruction input"));
            }
        }
    }
    if !frame_complete {
        return Err(failure("no completed JPEG coefficient frame"));
    }
    let recovered = capture
        .and_then(|capture| capture.take())
        .ok_or_else(|| failure("JPEG coefficient capture incomplete"))?;
    let boxes = boxes::extract(&decoder, input, options.limits.max_metadata_bytes)?;
    let jbrd = unique_box(&boxes, *b"jbrd")?
        .ok_or_else(|| failure("JXL has no original-JPEG reconstruction data"))?;
    let sources = ReconstructionMetadataSources {
        icc_profile: match decoder.embedded_color_profile() {
            Some(JxlColorProfile::Icc(bytes)) => Some(bytes),
            _ => None,
        },
        exif_box_payload: unique_box(&boxes, *b"Exif")?,
        xmp_box_payload: unique_box(&boxes, *b"xml ")?,
    };
    let mut limits = ReconstructionDecodeLimits {
        max_jbrd_bytes: options.limits.max_metadata_bytes,
        max_metadata_bytes: options.limits.max_metadata_bytes,
        max_coefficient_values: options
            .limits
            .max_samples
            .min(options.limits.max_output_bytes / 2),
        ..Default::default()
    };
    limits.write.max_output_bytes = options.limits.max_output_bytes;
    let components = recovered
        .components
        .into_iter()
        .map(|component| {
            Ok(CapturedJpegComponent {
                h_samp_factor: u32::from(component.horizontal_sampling_factor),
                v_samp_factor: u32::from(component.vertical_sampling_factor),
                width_in_blocks: component
                    .width_in_blocks
                    .try_into()
                    .map_err(|_| Error::Limit("JPEG block width"))?,
                height_in_blocks: component
                    .height_in_blocks
                    .try_into()
                    .map_err(|_| Error::Limit("JPEG block height"))?,
                coefficients: component.coefficients,
                quantization: component.quantization.map(i32::from),
            })
        })
        .collect::<Result<_>>()?;
    crate::jpeg::reconstruction_decode::reconstruct_jpeg(
        jbrd,
        CapturedJpegCoefficients {
            width: recovered.width,
            height: recovered.height,
            components,
        },
        sources,
        &limits,
    )
    .map_err(failure)
}
