use super::{
    AlphaMode, DecodeOptions, DecodedImage, backend_error, backend_options, boxes,
    buffers::Buffers, check_info, color, failure, runner::Runner,
};
use crate::{Error, Result};
use jxl_decoder::api::{Event, JxlDecoder, JxlFrameHeader, JxlOutputBuffer};

/// Decode every display frame, or every regular layer when coalescing is off.
pub fn decode(input: &[u8], options: &DecodeOptions) -> Result<DecodedImage> {
    decode_impl(input, None, options)
}

/// Decode a zero-based display frame (or raw layer in noncoalescing mode).
/// Required references are reconstructed sequentially; later data is not checked.
pub fn decode_frame(input: &[u8], index: usize, options: &DecodeOptions) -> Result<DecodedImage> {
    if index >= options.limits.max_frames {
        return Err(Error::Limit("requested frame index"));
    }
    decode_impl(input, Some(index), options)
}

fn decode_impl(
    mut input: &[u8],
    target: Option<usize>,
    options: &DecodeOptions,
) -> Result<DecodedImage> {
    let input_len = input.len();
    let mut backend = backend_options(input, options)?;
    backend.request_aux_boxes = boxes::REQUESTED.to_vec();
    if options.color.needs_float() {
        backend.alpha_mode = AlphaMode::AsIs;
    }
    let mut decoder = JxlDecoder::new(backend);
    decoder.close_input();
    let mut runner = Runner::new(options.threads)?;
    let mut buffers: Option<Buffers> = None;
    let mut header: Option<JxlFrameHeader> = None;
    let mut frames = Vec::new();
    let mut preview = None;
    let mut index = 0;
    let mut retained = 0usize;
    let mut complete = false;
    loop {
        let wanted = header
            .as_ref()
            .is_some_and(|h| h.is_preview || target.is_none_or(|n| n == index));
        let event = {
            let mut views: Vec<_> = buffers
                .as_mut()
                .map(|b| {
                    b.images
                        .iter_mut()
                        .map(|image| JxlOutputBuffer::from_image_rect_mut(image.as_rect_mut()))
                        .collect()
                })
                .unwrap_or_default();
            let output = (wanted && !views.is_empty()).then_some(views.as_mut_slice());
            decoder
                .process(&mut input, output, Some(&mut runner))
                .map_err(backend_error)?
        };
        match event {
            Event::BasicInfo => {
                let info = decoder
                    .basic_info()
                    .ok_or_else(|| failure("missing basic info"))?;
                check_info(info, options)?;
                let color_type = decoder
                    .current_pixel_format()
                    .ok_or_else(|| failure("missing pixel format"))?
                    .color_type;
                let format = Buffers::pixel_format(info, color_type, options);
                decoder.set_pixel_format(format).map_err(backend_error)?;
            }
            Event::FrameHeader => {
                let current = decoder
                    .frame_header()
                    .ok_or_else(|| failure("missing frame header"))?;
                if !current.is_preview && index >= options.limits.max_frames {
                    return Err(Error::Limit("decoded frame count"));
                }
                if current.is_preview || target.is_none_or(|n| n == index) {
                    if !buffers.as_ref().is_some_and(|b| b.has_size(current.size)) {
                        buffers = Some(Buffers::new(
                            current.size,
                            decoder.current_pixel_format().unwrap(),
                            options,
                        )?);
                    }
                    if options.allow_partial {
                        for plane in &mut buffers.as_mut().unwrap().images {
                            plane.fill_zero();
                        }
                    }
                    retained = retained
                        .checked_add(buffers.as_ref().unwrap().byte_count)
                        .ok_or(Error::Limit("output size overflow"))?;
                    if retained > options.limits.max_output_bytes {
                        return Err(Error::Limit("retained decoded outputs"));
                    }
                } else {
                    buffers = None;
                }
                header = Some(current);
            }
            Event::FrameComplete { .. } => {
                let current = header
                    .take()
                    .ok_or_else(|| failure("frame without header"))?;
                if let Some(allocated) = buffers.as_ref() {
                    let full_quality = decoder
                        .last_frame_progression()
                        .is_none_or(|p| p.full_quality);
                    let frame = allocated.copy_frame(index, &current, full_quality)?;
                    if current.is_preview {
                        preview = Some(frame);
                    } else {
                        frames
                            .try_reserve(1)
                            .map_err(|_| Error::Limit("decoded frame list"))?;
                        frames.push(frame);
                    }
                }
                if !current.is_preview {
                    if target == Some(index) {
                        break;
                    }
                    index += 1;
                }
            }
            Event::Complete => {
                complete = true;
                break;
            }
            Event::NeedMoreInput { .. } => {
                if !options.allow_partial {
                    return Err(failure("truncated JPEG XL input"));
                }
                if wanted && let (Some(current), Some(allocated)) = (&header, &mut buffers) {
                    let mut views: Vec<_> = allocated
                        .images
                        .iter_mut()
                        .map(|image| JxlOutputBuffer::from_image_rect_mut(image.as_rect_mut()))
                        .collect();
                    if decoder
                        .flush_pixels(&mut views, Some(&mut runner))
                        .map_err(backend_error)?
                    {
                        let frame = allocated.copy_frame(index, current, false)?;
                        if current.is_preview {
                            preview = Some(frame);
                        } else {
                            frames
                                .try_reserve(1)
                                .map_err(|_| Error::Limit("decoded frame list"))?;
                            frames.push(frame);
                        }
                    }
                }
                break;
            }
        }
    }
    if frames.is_empty() {
        return Err(failure(if target.is_some() {
            "requested frame is unavailable"
        } else {
            "JXL contains no decodable image frames"
        }));
    }
    let metadata = if complete {
        boxes::extract(&decoder, input, options.limits.max_metadata_bytes)?
    } else {
        Vec::new()
    };
    let output_complete = complete
        && frames.iter().all(|f| f.complete)
        && preview.as_ref().is_none_or(|f| f.complete);
    let mut image = DecodedImage {
        info: decoder
            .basic_info()
            .ok_or_else(|| failure("missing basic info"))?
            .clone(),
        frames,
        preview,
        embedded_profile: decoder.embedded_color_profile().cloned(),
        output_profile: decoder.output_color_profile().cloned(),
        output_intensity_target: decoder.basic_info().unwrap().tone_mapping.intensity_target,
        boxes: metadata,
        complete: output_complete,
        bytes_read: if complete && decoder.trailing_box().is_some() {
            input_len
        } else {
            input_len - input.len()
        },
    };
    image.output_intensity_target =
        color::process_image(&mut image, &options.color, &options.limits)?;
    super::convert::finish(&mut image, options)?;
    Ok(image)
}
