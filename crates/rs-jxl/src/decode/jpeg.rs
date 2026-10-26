//! Original-byte reconstruction from a complete JXL container, never pixels.
use super::{DecodeOptions, DecodedBox, backend_error, backend_options, boxes, failure, runner};
use crate::jpeg::data::JpegError;
use crate::jpeg::entropy_writer::WriteError;
use crate::jpeg::reconstruction_decode::{
    CapturedJpegCoefficients, CapturedJpegComponent, ReconstructionDecodeError,
    ReconstructionDecodeLimits, decode_reconstruction_metadata,
};
use crate::jpeg::reconstruction_metadata::ReconstructionMetadataSources;
use crate::jpeg::scan_writer::{ReconstructionError, reconstruct_with_pool};
use crate::{Error, Result};
use jxl_decoder::api::{
    Event, JxlColorProfile, JxlDecoder, JxlDecoderOptions, JxlJpegCoefficientCapture,
};

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

fn reconstruction_error(error: ReconstructionDecodeError) -> Error {
    match error {
        ReconstructionDecodeError::Limit(resource)
        | ReconstructionDecodeError::Structural(JpegError::LimitExceeded { resource, .. })
        | ReconstructionDecodeError::Reconstruction(ReconstructionError::Limit(resource))
        | ReconstructionDecodeError::Reconstruction(ReconstructionError::Structural(
            JpegError::LimitExceeded { resource, .. },
        )) => Error::Limit(resource),
        ReconstructionDecodeError::Reconstruction(ReconstructionError::Entropy(
            WriteError::OutputLimit { .. },
        )) => Error::Limit("JPEG reconstruction entropy output"),
        ReconstructionDecodeError::Reconstruction(ReconstructionError::Entropy(
            WriteError::AllocationFailed,
        )) => Error::Limit("JPEG reconstruction entropy allocation"),
        other => failure(other),
    }
}

/// Reconstruct the original JPEG byte stream from JXL coefficients and `jbrd`.
/// Requires a supported JPEG-transcode frame; pixel-only JXL is an error. No
/// lossy JPEG re-encoding fallback and no access to the original input JPEG.
/// Sample output format and orientation settings never alter JPEG bytes.
/// Supports multipass DCT8 capture and exact integer DC extra precision, with
/// JPEG RAW tables and JPEG-compatible CfL. Other transforms, non-RAW tables,
/// fractional/clamped DC, animations, LF frames, crops and extra channels are
/// rejected; an arbitrary pixel JXL is not an original-JPEG source.
pub fn reconstruct_jpeg(input: &[u8], options: &DecodeOptions) -> Result<Vec<u8>> {
    let _total = crate::profile::scope("decode.jpeg.total");
    let backend = prepare_backend(backend_options(input, options)?, options)?;
    let mut runner = runner::Runner::new(options.threads)?;
    reconstruct_with_runner(input, options, backend, &mut runner)
}

pub(super) fn prepare_backend(
    mut backend: JxlDecoderOptions,
    options: &DecodeOptions,
) -> Result<JxlDecoderOptions> {
    if options.limits.max_frames == 0 {
        return Err(Error::Limit("JPEG frame count"));
    }
    backend.adjust_orientation = false;
    backend.render_spot_colors = false;
    // Pixel preview/progression options must never select incomplete coefficients.
    backend.skip_preview = true;
    backend.max_downsampling = 1;
    backend.coalescing = true;
    backend.scan_frames_only = false;
    backend.request_aux_boxes = boxes::REQUESTED.to_vec();
    Ok(backend)
}

pub(super) fn reconstruct_with_runner(
    mut input: &[u8],
    options: &DecodeOptions,
    backend: JxlDecoderOptions,
    runner: &mut runner::Runner,
) -> Result<Vec<u8>> {
    let coefficient_stage = crate::profile::scope("decode.jpeg.coefficients");
    let mut decoder = JxlDecoder::new(backend);
    decoder.close_input();
    let mut capture: Option<JxlJpegCoefficientCapture> = None;
    let mut frame_complete = false;
    loop {
        match decoder
            .process(&mut input, None, Some(&mut *runner))
            .map_err(backend_error)?
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
                capture = Some(
                    decoder
                        .enable_jpeg_coefficient_capture()
                        .map_err(backend_error)?,
                );
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
    drop(coefficient_stage);
    let metadata_stage = crate::profile::scope("decode.jpeg.metadata");
    let boxes = boxes::extract(&decoder, input, options.limits.max_metadata_bytes)?;
    let jbrd = unique_box(&boxes, *b"jbrd")?
        .ok_or_else(|| failure("JXL has no original-JPEG reconstruction data"))?;
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
    let metadata = decode_reconstruction_metadata(jbrd, &limits).map_err(reconstruction_error)?;
    // Unreferenced display metadata is not an original-JPEG source. Multiple
    // boxes are ambiguous only when jbrd needs their bytes to restore an APP.
    let sources = ReconstructionMetadataSources {
        icc_profile: match decoder.embedded_color_profile() {
            Some(JxlColorProfile::Icc(bytes)) => Some(bytes),
            _ => None,
        },
        exif_box_payload: if metadata.needs_exif() {
            unique_box(&boxes, *b"Exif")?
        } else {
            None
        },
        xmp_box_payload: if metadata.needs_xmp() {
            unique_box(&boxes, *b"xml ")?
        } else {
            None
        },
    };
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
    drop(metadata_stage);
    let model_stage = crate::profile::scope("decode.jpeg.writer_model");
    let jpeg = metadata
        .into_jpeg(
            CapturedJpegCoefficients {
                width: recovered.width,
                height: recovered.height,
                components,
            },
            sources,
            &limits,
        )
        .map_err(reconstruction_error)?;
    drop(model_stage);
    reconstruct_with_pool(&jpeg, &limits.write, runner.pool())
        .map_err(ReconstructionDecodeError::Reconstruction)
        .map_err(reconstruction_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruction_resource_errors_retain_their_public_classification() {
        for error in [
            ReconstructionDecodeError::Limit("expanded metadata"),
            ReconstructionDecodeError::Structural(JpegError::LimitExceeded {
                offset: 3,
                resource: "records",
            }),
            ReconstructionDecodeError::Reconstruction(ReconstructionError::Limit("scan work")),
            ReconstructionDecodeError::Reconstruction(ReconstructionError::Structural(
                JpegError::LimitExceeded {
                    offset: 7,
                    resource: "tables",
                },
            )),
            ReconstructionDecodeError::Reconstruction(ReconstructionError::Entropy(
                WriteError::OutputLimit { limit: 1 },
            )),
            ReconstructionDecodeError::Reconstruction(ReconstructionError::Entropy(
                WriteError::AllocationFailed,
            )),
        ] {
            assert!(matches!(reconstruction_error(error), Error::Limit(_)));
        }
        for error in [
            ReconstructionDecodeError::Truncated("Brotli stream"),
            ReconstructionDecodeError::MissingSource("Exif box"),
            ReconstructionDecodeError::Unsupported("JPEG marker"),
            ReconstructionDecodeError::Structural(JpegError::Invalid {
                offset: 0,
                reason: "marker",
            }),
            ReconstructionDecodeError::Reconstruction(ReconstructionError::Entropy(
                WriteError::MissingHuffmanSymbol(1),
            )),
        ] {
            assert!(matches!(reconstruction_error(error), Error::Decoding(_)));
        }
    }
}
