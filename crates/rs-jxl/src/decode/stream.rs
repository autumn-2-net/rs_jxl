//! Persistent push/EOF and blocking-reader adapters for the maintained decoder.
//!
//! Input slices are borrowed while processing. Only unconsumed input and an
//! open-ended requested metadata box are retained here; the backend can also
//! buffer frame sections, references and out-of-order container chunks.

use std::io::{ErrorKind, Read};

use jxl_decoder::api::{Event, JxlDecoder, JxlFrameHeader, JxlNativeIntegerCapture};

use super::{
    AlphaMode, DecodeOptions, DecodedBox, DecodedFrame, DecodedImage, JxlBasicInfo,
    JxlColorProfile, backend_error, backend_options, boxes, buffers::Buffers, check_info, color,
    convert, failure, native, runner::Runner,
};
use crate::{Error, Result};

const DEFAULT_CHUNK_BYTES: usize = 64 * 1024;

/// File-level results available only after EOF. Pixels are returned separately
/// by `StreamDecoder::take_frames` or a reader callback.
pub struct StreamSummary {
    pub info: JxlBasicInfo,
    pub embedded_profile: Option<JxlColorProfile>,
    pub output_profile: Option<JxlColorProfile>,
    pub output_intensity_target: f32,
    pub boxes: Vec<DecodedBox>,
    /// Excludes preview, and counts returned raw layers when coalescing is off.
    pub frame_count: usize,
    pub has_preview: bool,
    /// False for opt-in partial or progressive output.
    pub complete: bool,
    /// Bytes consumed by the parser, including read-ahead and trailing metadata.
    /// May be less than bytes supplied for a bare codestream with ignored suffix.
    pub bytes_read: usize,
}

/// Incremental pixel decoder. Each instance owns one persistent backend and
/// thread pool. `push` is not EOF; call `finish` exactly once to validate EOF.
///
/// Drain `take_frames` after every push, and again after finish (which may flush
/// partial pixels). Draining never resets cumulative input/output/frame limits.
/// Any error poisons the instance; already queued frames remain available, but
/// they do not certify that the rest of the file is valid.
pub struct StreamDecoder {
    options: DecodeOptions,
    decoder: JxlDecoder,
    runner: Runner,
    buffers: Option<Buffers>,
    native_capture: Option<JxlNativeIntegerCapture>,
    header: Option<JxlFrameHeader>,
    pending: Vec<u8>,
    trailing: Vec<u8>,
    ready: Vec<DecodedFrame>,
    input_bytes: usize,
    consumed_bytes: usize,
    reserved_output_bytes: usize,
    returned_output_bytes: usize,
    frame_count: usize,
    has_preview: bool,
    all_frames_complete: bool,
    output_profile: Option<JxlColorProfile>,
    output_intensity_target: Option<f32>,
    backend_complete: bool,
    finished: bool,
    failed: bool,
}

impl StreamDecoder {
    pub fn new(options: &DecodeOptions) -> Result<Self> {
        let mut backend = backend_options(&[], options)?;
        backend.request_aux_boxes = boxes::REQUESTED.to_vec();
        if options.color.needs_float() {
            backend.alpha_mode = AlphaMode::AsIs;
        }
        Ok(Self {
            options: options.clone(),
            decoder: JxlDecoder::new(backend),
            runner: Runner::new(options.threads)?,
            buffers: None,
            native_capture: None,
            header: None,
            pending: Vec::new(),
            trailing: Vec::new(),
            ready: Vec::new(),
            input_bytes: 0,
            consumed_bytes: 0,
            reserved_output_bytes: 0,
            returned_output_bytes: 0,
            frame_count: 0,
            has_preview: false,
            all_frames_complete: true,
            output_profile: None,
            output_intensity_target: None,
            backend_complete: false,
            finished: false,
            failed: false,
        })
    }

    /// Supply another chunk. Empty chunks are allowed and do not close input.
    /// Frames completed before an error remain accessible with `take_frames`.
    pub fn push(&mut self, input: &[u8]) -> Result<()> {
        self.check_open()?;
        let result = self.push_inner(input);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Transfer completed frames without copying pixels. Preview, when enabled,
    /// is identified by `frame.header.is_preview` and does not consume index 0.
    /// This remains callable after EOF or an error.
    pub fn take_frames(&mut self) -> Vec<DecodedFrame> {
        std::mem::take(&mut self.ready)
    }

    /// Parsed source info, if the header has already arrived. Pixel profiles may
    /// still change during the opt-in color processing of a completed frame.
    pub fn basic_info(&self) -> Option<&JxlBasicInfo> {
        self.decoder.basic_info()
    }

    /// Signal EOF, validate trailing boxes, and optionally flush partial pixels.
    /// A complete frame emitted earlier does not make this call optional.
    pub fn finish(&mut self) -> Result<StreamSummary> {
        self.check_open()?;
        self.finished = true;
        let result = self.finish_inner();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn check_open(&self) -> Result<()> {
        if self.failed || self.finished {
            return Err(Error::InvalidInput(
                "stream decoder is already finished or failed".into(),
            ));
        }
        Ok(())
    }

    fn push_inner(&mut self, input: &[u8]) -> Result<()> {
        self.input_bytes = self
            .input_bytes
            .checked_add(input.len())
            .filter(|&n| n <= self.options.limits.max_input_bytes)
            .ok_or(Error::Limit("JXL input bytes"))?;
        if self.backend_complete {
            return self.retain_trailing(input);
        }
        if self.pending.is_empty() {
            let mut remaining = input;
            self.drive(&mut remaining)?;
            self.retain_remaining(remaining)
        } else {
            // Normally the backend consumes an exhausted chunk into its own
            // resumable buffers. Retain a suffix if a stage leaves one unread.
            let mut pending = std::mem::take(&mut self.pending);
            append(&mut pending, input, "pending stream input")?;
            let mut remaining = pending.as_slice();
            self.drive(&mut remaining)?;
            self.retain_remaining(remaining)
        }
    }

    fn retain_remaining(&mut self, input: &[u8]) -> Result<()> {
        if self.backend_complete {
            self.retain_trailing(input)
        } else {
            append(&mut self.pending, input, "pending stream input")
        }
    }

    fn retain_trailing(&mut self, input: &[u8]) -> Result<()> {
        let Some(entry) = self.decoder.trailing_box() else {
            // Same policy as one-shot decoding: the raw codestream (or an
            // unrequested size-zero box) can finish before the enclosing input.
            return Ok(());
        };
        if !entry.is_compressed() {
            entry
                .raw_data()
                .len()
                .checked_add(self.trailing.len())
                .and_then(|n| n.checked_add(input.len()))
                .filter(|&n| n <= self.options.limits.max_metadata_bytes)
                .ok_or(Error::Limit("decompressed metadata bytes"))?;
        }
        append(&mut self.trailing, input, "trailing stream metadata")
    }

    fn drive(&mut self, input: &mut &[u8]) -> Result<()> {
        while !self.backend_complete {
            let before = input.len();
            let event = {
                let mut views: Vec<_> = self
                    .buffers
                    .as_mut()
                    .filter(|_| self.header.is_some())
                    .map(Buffers::views)
                    .transpose()?
                    .unwrap_or_default();
                let output = (!views.is_empty()).then_some(views.as_mut_slice());
                self.decoder.process(input, output, Some(&mut self.runner))
            };
            self.consumed_bytes = self
                .consumed_bytes
                .checked_add(before - input.len())
                .ok_or(Error::Limit("stream input counter"))?;
            match event.map_err(backend_error)? {
                Event::BasicInfo => {
                    let info = self
                        .decoder
                        .basic_info()
                        .ok_or_else(|| failure("missing basic info"))?;
                    check_info(info, &self.options)?;
                    let color_type = self
                        .decoder
                        .current_pixel_format()
                        .ok_or_else(|| failure("missing pixel format"))?
                        .color_type;
                    if native::needed(info, &self.options)? {
                        self.native_capture = Some(
                            self.decoder
                                .enable_native_integer_capture()
                                .map_err(backend_error)?,
                        );
                    } else {
                        let format = Buffers::pixel_format(info, color_type, &self.options);
                        self.decoder
                            .set_pixel_format(format)
                            .map_err(backend_error)?;
                    }
                }
                Event::FrameHeader => self.begin_frame()?,
                Event::FrameComplete { .. } => {
                    let header = self
                        .header
                        .take()
                        .ok_or_else(|| failure("frame without header"))?;
                    let complete = self
                        .decoder
                        .last_frame_progression()
                        .is_none_or(|p| p.full_quality);
                    let mut frame = if let Some(capture) = &self.native_capture {
                        let captured = capture
                            .take()
                            .ok_or_else(|| failure("missing exact native integer frame"))?;
                        native::frame(
                            captured,
                            self.decoder.basic_info().unwrap(),
                            &header,
                            self.frame_count,
                            &self.options,
                        )?
                    } else {
                        self.buffers
                            .take()
                            .ok_or_else(|| failure("frame without output buffers"))?
                            .into_frame(self.frame_count, &header, complete)?
                    };
                    frame.complete = complete;
                    self.publish(frame)?;
                }
                Event::Complete => self.backend_complete = true,
                Event::NeedMoreInput { .. } => return Ok(()),
            }
        }
        Ok(())
    }

    fn begin_frame(&mut self) -> Result<()> {
        let header = self
            .decoder
            .frame_header()
            .ok_or_else(|| failure("missing frame header"))?;
        if !header.is_preview && self.frame_count >= self.options.limits.max_frames {
            return Err(Error::Limit("decoded frame count"));
        }
        if self.native_capture.is_some() {
            let color = if self
                .decoder
                .current_pixel_format()
                .unwrap()
                .color_type
                .is_grayscale()
            {
                1
            } else {
                3
            };
            let bytes = native::byte_count(
                self.decoder.basic_info().unwrap(),
                color,
                header.size,
                &self.options,
            )?;
            self.reserve_output(bytes)?;
            self.header = Some(header);
            return Ok(());
        }
        self.buffers = Some(Buffers::new(
            header.size,
            self.decoder
                .current_pixel_format()
                .ok_or_else(|| failure("missing pixel format"))?,
            &self.options,
        )?);
        let byte_count = self.buffers.as_ref().unwrap().byte_count;
        self.reserve_output(byte_count)?;
        self.header = Some(header);
        Ok(())
    }

    fn reserve_output(&mut self, byte_count: usize) -> Result<()> {
        self.reserved_output_bytes = self
            .reserved_output_bytes
            .checked_add(byte_count)
            .filter(|&n| n <= self.options.limits.max_output_bytes)
            .ok_or(Error::Limit("retained decoded outputs"))?;
        Ok(())
    }

    fn publish(&mut self, frame: DecodedFrame) -> Result<()> {
        let is_preview = frame.header.is_preview;
        let info = self
            .decoder
            .basic_info()
            .ok_or_else(|| failure("missing basic info"))?
            .clone();
        let intensity = info.tone_mapping.intensity_target;
        // Reuse the ordinary conversion pipeline, with a single frame instead
        // of retaining every frame until EOF. Never reprocess previous outputs.
        let mut image = DecodedImage {
            info,
            frames: Vec::new(),
            preview: None,
            embedded_profile: self.decoder.embedded_color_profile().cloned(),
            output_profile: self.decoder.output_color_profile().cloned(),
            output_intensity_target: intensity,
            boxes: Vec::new(),
            complete: frame.complete,
            bytes_read: self.consumed_bytes,
        };
        if is_preview {
            image.preview = Some(frame);
        } else {
            image
                .frames
                .try_reserve_exact(1)
                .map_err(|_| Error::Limit("decoded frame list"))?;
            image.frames.push(frame);
        }
        image.output_intensity_target =
            color::process_image(&mut image, &self.options.color, &self.options.limits)?;
        convert::finish(&mut image, &self.options)?;
        let frame = if is_preview {
            image.preview.take().unwrap()
        } else {
            image.frames.pop().unwrap()
        };
        let bytes = std::iter::once(&frame.pixels)
            .chain(&frame.extra_planes)
            .try_fold(0usize, |n, plane| {
                plane
                    .len()
                    .checked_mul(usize::from(plane.bits()) / 8)
                    .and_then(|bytes| n.checked_add(bytes))
            })
            .ok_or(Error::Limit("output size overflow"))?;
        self.returned_output_bytes = self
            .returned_output_bytes
            .checked_add(bytes)
            .filter(|&n| n <= self.options.limits.max_output_bytes)
            .ok_or(Error::Limit("retained decoded outputs"))?;
        self.ready
            .try_reserve(1)
            .map_err(|_| Error::Limit("decoded frame list"))?;
        self.all_frames_complete &= frame.complete;
        self.has_preview |= is_preview;
        self.frame_count += usize::from(!is_preview);
        self.output_profile = image.output_profile;
        self.output_intensity_target = Some(image.output_intensity_target);
        self.ready.push(frame);
        Ok(())
    }

    fn finish_inner(&mut self) -> Result<StreamSummary> {
        self.decoder.close_input();
        if !self.backend_complete {
            let pending = std::mem::take(&mut self.pending);
            let mut remaining = pending.as_slice();
            self.drive(&mut remaining)?;
            self.retain_remaining(remaining)?;
        }
        if !self.backend_complete {
            if !self.options.allow_partial {
                return Err(failure("truncated JPEG XL input"));
            }
            self.flush_partial()?;
        }
        if self.frame_count == 0 {
            return Err(failure("JXL contains no decodable image frames"));
        }
        let boxes = if self.backend_complete {
            boxes::extract(
                &self.decoder,
                &self.trailing,
                self.options.limits.max_metadata_bytes,
            )?
        } else {
            Vec::new()
        };
        Ok(StreamSummary {
            info: self
                .decoder
                .basic_info()
                .ok_or_else(|| failure("missing basic info"))?
                .clone(),
            embedded_profile: self.decoder.embedded_color_profile().cloned(),
            output_profile: self.output_profile.clone(),
            output_intensity_target: self.output_intensity_target.unwrap(),
            boxes,
            frame_count: self.frame_count,
            has_preview: self.has_preview,
            complete: self.backend_complete && self.all_frames_complete,
            bytes_read: if self.backend_complete && self.decoder.trailing_box().is_some() {
                self.input_bytes
            } else {
                self.consumed_bytes
            },
        })
    }

    fn flush_partial(&mut self) -> Result<()> {
        let (Some(header), Some(buffers)) = (&self.header, &mut self.buffers) else {
            return Ok(());
        };
        let flushed = {
            let mut views = buffers.views()?;
            self.decoder
                .flush_pixels(&mut views, Some(&mut self.runner))
                .map_err(backend_error)?
        };
        if flushed {
            let frame = buffers.copy_frame(self.frame_count, header, false)?;
            self.publish(frame)?;
        }
        self.header = None;
        Ok(())
    }
}

#[cfg(test)]
#[path = "stream_limit_tests.rs"]
mod limit_tests;

fn append(output: &mut Vec<u8>, input: &[u8], allocation: &'static str) -> Result<()> {
    output
        .try_reserve(input.len())
        .map_err(|_| Error::Limit(allocation))?;
    output.extend_from_slice(input);
    Ok(())
}

/// Decode a blocking reader using a reusable 64 KiB input buffer. Pixel outputs
/// are collected, but decoding starts without waiting for the whole input.
pub fn decode_reader(reader: impl Read, options: &DecodeOptions) -> Result<DecodedImage> {
    decode_reader_with_chunk_size(reader, options, DEFAULT_CHUNK_BYTES)
}

/// Like `decode_reader`, with a positive maximum input read size. The actual
/// allocation is capped by the cumulative input limit (plus one EOF probe byte).
pub fn decode_reader_with_chunk_size(
    reader: impl Read,
    options: &DecodeOptions,
    chunk_bytes: usize,
) -> Result<DecodedImage> {
    let mut frames = Vec::new();
    let mut preview = None;
    let summary = decode_reader_with(reader, options, chunk_bytes, |frame| {
        if frame.header.is_preview {
            preview = Some(frame);
        } else {
            frames
                .try_reserve(1)
                .map_err(|_| Error::Limit("decoded frame list"))?;
            frames.push(frame);
        }
        Ok(())
    })?;
    Ok(DecodedImage {
        info: summary.info,
        frames,
        preview,
        embedded_profile: summary.embedded_profile,
        output_profile: summary.output_profile,
        output_intensity_target: summary.output_intensity_target,
        boxes: summary.boxes,
        complete: summary.complete,
        bytes_read: summary.bytes_read,
    })
}

/// Emit pixels between reader calls, before EOF or later parse/I/O errors.
/// Callbacks receive preview and regular frames; inspect `header.is_preview`.
/// Errors from the reader or callback are not treated as partial-input EOF.
/// `Interrupted` reads are retried; `WouldBlock` is returned to the caller.
/// Use `StreamDecoder` directly for nonblocking or externally polled input.
pub fn decode_reader_with(
    mut reader: impl Read,
    options: &DecodeOptions,
    chunk_bytes: usize,
    mut emit: impl FnMut(DecodedFrame) -> Result<()>,
) -> Result<StreamSummary> {
    if chunk_bytes == 0 {
        return Err(Error::InvalidInput(
            "input chunk size must be positive".into(),
        ));
    }
    let mut stream = StreamDecoder::new(options)?;
    let capacity = chunk_bytes.min(options.limits.max_input_bytes.saturating_add(1));
    let mut input = Vec::new();
    input
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Limit("stream reader buffer"))?;
    input.resize(capacity, 0);
    loop {
        // At the limit, read at most one byte to distinguish exact EOF from an
        // oversized input. Never allocate or accept a backend size_hint blindly.
        let remaining = options.limits.max_input_bytes - stream.input_bytes;
        let count = input.len().min(remaining.saturating_add(1));
        let read = match reader.read(&mut input[..count]) {
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            result => result?,
        };
        if read == 0 {
            let result = stream.finish();
            for frame in stream.take_frames() {
                emit(frame)?;
            }
            return result;
        }
        let result = stream.push(&input[..read]);
        for frame in stream.take_frames() {
            emit(frame)?;
        }
        result?;
    }
}
