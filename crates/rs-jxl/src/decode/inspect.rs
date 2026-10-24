use super::{
    DecodeOptions, DecodedBox, JxlBasicInfo, JxlColorProfile, JxlFrameHeader, backend_error,
    backend_options, boxes, buffers::Buffers, check_info, failure,
};
use crate::{Error, Result};
use jxl_decoder::api::{Event, JxlDecoder};

pub struct ImageInspection {
    pub info: JxlBasicInfo,
    pub frames: Vec<JxlFrameHeader>,
    pub preview: Option<JxlFrameHeader>,
    pub embedded_profile: Option<JxlColorProfile>,
    /// The decoder's default pixel profile, not a requested CMS conversion.
    pub output_profile: Option<JxlColorProfile>,
    pub boxes: Vec<DecodedBox>,
    pub bytes_read: usize,
}

/// Parse image/color headers without decoding pixels or validating the remainder.
pub fn read_basic_info(mut input: &[u8], options: &DecodeOptions) -> Result<JxlBasicInfo> {
    let mut decoder = JxlDecoder::new(backend_options(input, options)?);
    decoder.close_input();
    match decoder
        .process(&mut input, None, None)
        .map_err(backend_error)?
    {
        Event::BasicInfo => {
            let info = decoder
                .basic_info()
                .ok_or_else(|| failure("missing basic info"))?;
            check_info(info, options)?;
            Ok(info.clone())
        }
        _ => Err(failure("incomplete JPEG XL image header")),
    }
}

/// Scan frame headers and metadata without decoding frame sections or references.
/// This validates container/frame structure, not the skipped compressed pixels.
/// The entire file must be present even when `allow_partial` is set.
pub fn inspect(mut input: &[u8], options: &DecodeOptions) -> Result<ImageInspection> {
    let input_len = input.len();
    let mut backend = backend_options(input, options)?;
    backend.scan_frames_only = true;
    backend.request_aux_boxes = boxes::REQUESTED.to_vec();
    let mut decoder = JxlDecoder::new(backend);
    decoder.close_input();
    let mut frames = Vec::new();
    let mut preview = None;
    loop {
        match decoder
            .process(&mut input, None, None)
            .map_err(backend_error)?
        {
            Event::BasicInfo => {
                let info = decoder.basic_info().unwrap();
                check_info(info, options)?;
                let color_type = decoder.current_pixel_format().unwrap().color_type;
                let format = Buffers::pixel_format(info, color_type, options);
                decoder.set_pixel_format(format).map_err(backend_error)?;
            }
            Event::FrameHeader => {
                let header = decoder
                    .frame_header()
                    .ok_or_else(|| failure("missing frame header"))?;
                if header.is_preview {
                    preview = Some(header);
                } else {
                    if frames.len() >= options.limits.max_frames {
                        return Err(Error::Limit("frame header count"));
                    }
                    frames
                        .try_reserve(1)
                        .map_err(|_| Error::Limit("frame header allocation"))?;
                    frames.push(header);
                }
            }
            Event::FrameComplete { .. } => {}
            Event::Complete => break,
            Event::NeedMoreInput { .. } => return Err(failure("truncated JPEG XL input")),
        }
    }
    let metadata = boxes::extract(&decoder, input, options.limits.max_metadata_bytes)?;
    Ok(ImageInspection {
        info: decoder
            .basic_info()
            .ok_or_else(|| failure("missing basic info"))?
            .clone(),
        frames,
        preview,
        embedded_profile: decoder.embedded_color_profile().cloned(),
        output_profile: decoder.output_color_profile().cloned(),
        boxes: metadata,
        bytes_read: if decoder.trailing_box().is_some() {
            input_len
        } else {
            input_len - input.len()
        },
    })
}
