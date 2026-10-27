//! Explicit original-size layers on a shared canvas, without padding samples.
//! Geometry and metadata are native-owned policy; frame coding uses the same
//! standard Modular encoder as equal-size multiframe input.
use crate::{
    EncodeOptions, Error, Image, Result, codestream, encode as encoder, execution,
    frame::{BlendInfo, Frame, FrameOptions},
    metadata::Animation,
    region_input::ImageDescriptor,
};
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct LayerOptions {
    /// None infers the maximum positive extent. Negative origins require an
    /// explicit canvas, whose bounds clip only composited output, not raw layers.
    pub canvas: Option<(u32, u32)>,
    /// Empty uses independent, zero-duration REPLACE layers. Otherwise one entry
    /// per image is required. Native FrameOptions has explicit reference semantics.
    pub frames: Vec<FrameOptions>,
    /// None infers default animation timing only when durations/timecodes need it.
    pub animation: Option<Animation>,
}

/// Slot 1 starts as a zero background and these defaults never overwrite it.
/// Nonfinal zero-duration layers save slot 0, so source=0 explicitly opts into
/// reusing their composited state. No synthetic full-canvas image is encoded.
pub fn default_frame_options() -> FrameOptions {
    FrameOptions {
        blend: BlendInfo {
            source: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

pub struct LayerPlan<'a> {
    pub canvas: ImageDescriptor,
    pub frames: Vec<Frame<'a>>,
}

#[derive(Clone, Debug)]
pub struct LayerInfo {
    pub width: u32,
    pub height: u32,
    pub options: FrameOptions,
}

#[derive(Debug)]
pub struct LayerEncodeResult {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// One regular, non-reference-only layer per input image.
    pub layers: Vec<LayerInfo>,
    /// Planning and coding, excluding pool construction and adapter transport.
    pub codec_seconds: f64,
}

/// Infer only canvas geometry and timing. Samples, channels, colors, depths,
/// original layer rectangles and the order of images remain unchanged.
pub fn plan<'a>(images: &'a [Image], options: &LayerOptions) -> Result<LayerPlan<'a>> {
    let first = images
        .first()
        .ok_or_else(|| Error::InvalidInput("layers require nonempty images".into()))?;
    if !options.frames.is_empty() && options.frames.len() != images.len() {
        return Err(Error::InvalidInput(
            "layer options must match image count".into(),
        ));
    }
    let mut frames = Vec::new();
    frames
        .try_reserve_exact(images.len())
        .map_err(|_| Error::Limit("layer descriptors"))?;
    let (mut width, mut height) = (0u32, 0u32);
    let (mut timed, mut timecodes) = (false, false);
    for (index, image) in images.iter().enumerate() {
        let settings = options
            .frames
            .get(index)
            .cloned()
            .unwrap_or_else(default_frame_options);
        if settings.reference_only {
            return Err(Error::InvalidInput(
                "encode_layers requires regular layers; use encode_frames for reference-only input"
                    .into(),
            ));
        }
        if options.canvas.is_none() {
            if settings.origin.0 < 0 || settings.origin.1 < 0 {
                return Err(Error::InvalidInput(
                    "negative layer origins require an explicit canvas".into(),
                ));
            }
            let right = (settings.origin.0 as u32)
                .checked_add(image.width())
                .ok_or_else(|| Error::InvalidInput("layer horizontal extent overflows".into()))?;
            let bottom = (settings.origin.1 as u32)
                .checked_add(image.height())
                .ok_or_else(|| Error::InvalidInput("layer vertical extent overflows".into()))?;
            width = width.max(right);
            height = height.max(bottom);
        }
        timed |= settings.duration != 0;
        timecodes |= settings.timecode != 0;
        frames.push(Frame {
            image,
            options: settings,
        });
    }
    let (width, height) = options.canvas.unwrap_or((width, height));
    let mut metadata = first.metadata().clone();
    metadata.animation = options.animation.or_else(|| {
        (timed || timecodes).then_some(Animation {
            timecodes,
            ..Default::default()
        })
    });
    let canvas = ImageDescriptor::new(width, height, first.layout(), metadata)?;
    // Validate all metadata/reference semantics before any encoder work. Input
    // animation metadata alone is replaced by the sequence's explicit timing.
    encoder::validate_frame_sequence(
        canvas.geometry(),
        canvas.layout(),
        &frames,
        &EncodeOptions::default(),
    )?;
    Ok(LayerPlan { canvas, frames })
}

pub fn encode(
    images: &[Image],
    options: &EncodeOptions,
    layers: &LayerOptions,
) -> Result<LayerEncodeResult> {
    if options.threads == 1 || execution::is_serial() {
        return encode_in_pool(images, options, layers, None);
    }
    crate::codec::Encoder::new(options.threads)?.encode_layers(images, options, layers)
}

pub(crate) fn encode_in_pool(
    images: &[Image],
    options: &EncodeOptions,
    layers: &LayerOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<LayerEncodeResult> {
    let work = || encode_inner(images, options, layers);
    match pool {
        Some(pool) => execution::with_pool(pool, work),
        None => execution::serial(work),
    }
}

fn encode_inner(
    images: &[Image],
    options: &EncodeOptions,
    layers: &LayerOptions,
) -> Result<LayerEncodeResult> {
    let started = Instant::now();
    let plan = plan(images, layers)?;
    let canvas = plan.canvas.geometry();
    encoder::validate_geometry(canvas, options)?;
    encoder::validate_frame_sequence(canvas, plan.canvas.layout(), &plan.frames, options)?;
    let mut stream = codestream::image_header_fields(
        canvas.width,
        canvas.height,
        canvas.metadata,
        options.max_output_bytes,
    )?;
    // A raw layer may extend beyond a smaller canvas. Its dimensions also
    // participate in the Level 5/10 decision, without allocating a canvas image.
    let mut level_geometry = canvas;
    for frame in &plan.frames {
        if !codestream::container_prefix_geometry(frame.image.into(), 0)?.is_empty() {
            level_geometry = frame.image.into();
            break;
        }
    }
    let mut info = Vec::new();
    info.try_reserve_exact(plan.frames.len())
        .map_err(|_| Error::Limit("layer reports"))?;
    for (index, frame) in plan.frames.iter().enumerate() {
        let mut bounded = options.clone();
        bounded.max_output_bytes = options
            .max_output_bytes
            .checked_sub(stream.len())
            .filter(|&n| n > 0)
            .ok_or(Error::Limit("encoded output bytes"))?;
        let (bytes, _) = encoder::encode_logical_frame_geometry(
            canvas,
            frame,
            &bounded,
            index + 1 == plan.frames.len(),
        )?;
        let length = stream
            .len()
            .checked_add(bytes.len())
            .filter(|&n| n <= options.max_output_bytes)
            .ok_or(Error::Limit("encoded output bytes"))?;
        stream
            .try_reserve(length - stream.len())
            .map_err(|_| Error::Limit("layer output allocation"))?;
        stream.extend_from_slice(&bytes);
        info.push(LayerInfo {
            width: frame.image.width(),
            height: frame.image.height(),
            options: frame.options.clone(),
        });
    }
    let mut prefix = codestream::container_prefix_geometry(level_geometry, stream.len())?;
    let length = prefix
        .len()
        .checked_add(stream.len())
        .filter(|&n| n <= options.max_output_bytes)
        .ok_or(Error::Limit("encoded output bytes"))?;
    let bytes = if prefix.is_empty() {
        stream
    } else {
        prefix
            .try_reserve(length - prefix.len())
            .map_err(|_| Error::Limit("layer container allocation"))?;
        prefix.extend_from_slice(&stream);
        prefix
    };
    Ok(LayerEncodeResult {
        bytes,
        width: canvas.width,
        height: canvas.height,
        layers: info,
        codec_seconds: started.elapsed().as_secs_f64(),
    })
}
