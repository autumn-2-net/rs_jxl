// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Synchronous chunked-frame input and ordered output: encode.cc AddChunkedFrame,
//! enc_frame.cc EncodeFrameStreaming/EncodeFrameOneShot at b866b3d.

use super::{seek_output::SeekOutput, seekable::AreaFrame};
use crate::frame::{Frame, FrameOptions};
use crate::modular::orchestration::{GroupwiseDecision, groupwise_decision_geometry};
use crate::region_input::{self, FrameAreaPlan, ImageDescriptor, RegionSource};
use crate::{EncodeOptions, Error, Result, codestream, execution};
use std::io::{Seek, Write};

mod index;
use index::IndexState;

/// Bounded selected-entry metadata, not a list of all frames or pixel buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameIndexOptions {
    pub max_entries: usize,
    pub max_payload_bytes: usize,
}

impl Default for FrameIndexOptions {
    fn default() -> Self {
        Self {
            max_entries: 4096,
            max_payload_bytes: 1024 * 1024,
        }
    }
}

/// Policy for frames which fail the frozen source's mode-1 eligibility gates.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FrameBuffering {
    /// Reject before input/output callbacks. Never stage a whole input frame.
    #[default]
    StrictAreas,
    /// Permit source one-shot encoding of only the submitted frame. The limit
    /// covers sample staging, not codec scratch or compressed output.
    CurrentFrame { max_sample_bytes: usize },
}

/// Stream-wide resources, fixed before the first frame. Per-frame EncodeOptions
/// must use the same thread count; their output cap additionally limits that frame.
#[derive(Clone, Debug)]
pub struct FrameEncoderOptions {
    pub threads: usize,
    pub max_output_bytes: usize,
    pub buffering: FrameBuffering,
    /// Select the first frame, and each later keyframe, with write_indexed_frame.
    pub index: Option<FrameIndexOptions>,
}

impl Default for FrameEncoderOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            max_output_bytes: 512 * 1024 * 1024,
            buffering: FrameBuffering::StrictAreas,
            index: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameProgressMode {
    FrameAreas,
    CurrentFrameBuffered,
}

/// Returned only after the submitted logical frame's directory and box length
/// are finalized. All byte positions are relative to the sink's initial position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameProgress {
    pub frame_index: u32,
    pub mode: FrameProgressMode,
    pub bytes_written: usize,
    pub finalized_bytes: usize,
    pub is_last: bool,
}

/// Push one region provider at a time, without retaining it or looking ahead.
/// The canvas owns metadata only. Output is ordered version-0 jxlp, Level 10
/// from the start, so later crops cannot invalidate the declared container level.
/// Neither the provider nor the sink needs Send; codec work uses one private pool.
pub struct FrameEncoder<W> {
    canvas: ImageDescriptor,
    config: FrameEncoderOptions,
    writer: SeekOutput<W>,
    image_header: Vec<u8>,
    pool: Option<rayon::ThreadPool>,
    references: [Option<bool>; 4],
    next_frame: u32,
    complete: bool,
    poisoned: bool,
    started: bool,
    codestream_bytes: u64,
    index: Option<IndexState>,
    index_written: bool,
}

impl<W: Write + Seek> FrameEncoder<W> {
    /// Validate stream metadata/resources and create its sole pool. Does not
    /// write bytes (it queries the sink position). The first push also validates
    /// frame metadata, references, buffering eligibility and limits before output.
    pub fn new(canvas: ImageDescriptor, config: FrameEncoderOptions, output: W) -> Result<Self> {
        canvas.validate()?;
        if !(1..=256).contains(&config.threads) {
            return Err(Error::InvalidInput("stream threads must be 1..256".into()));
        }
        if matches!(
            config.buffering,
            FrameBuffering::CurrentFrame {
                max_sample_bytes: 0
            }
        ) {
            return Err(Error::InvalidInput(
                "buffered sample limit must be positive".into(),
            ));
        }
        let image_header = execution::serial(|| {
            codestream::image_header_fields(
                canvas.width(),
                canvas.height(),
                canvas.metadata(),
                config.max_output_bytes,
            )
        })?;
        let index = config
            .index
            .map(|index| IndexState::new(index, canvas.geometry()))
            .transpose()?;
        let index_minimum = if index.is_some() { 16 + 12 } else { 0 };
        // Signature/ftyp, jxll and the extended jxlp header including its counter.
        crate::container::PREFIX
            .len()
            .checked_add(9 + 20 + index_minimum)
            .and_then(|n| n.checked_add(image_header.len()))
            .filter(|&n| n < config.max_output_bytes)
            .ok_or(Error::Limit("streaming output bytes"))?;
        let pool = if config.threads == 1 || execution::is_serial() {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.threads)
                    .build()
                    .map_err(|e| Error::Encoding(e.to_string()))?,
            )
        };
        let writer = SeekOutput::new(output, config.max_output_bytes)?;
        let codestream_bytes = image_header.len() as u64;
        Ok(Self {
            canvas,
            config,
            writer,
            image_header,
            pool,
            references: [None; 4],
            next_frame: 0,
            complete: false,
            poisoned: false,
            started: false,
            codestream_bytes,
            index,
            index_written: false,
        })
    }

    /// Synchronously consume/release this frame's regions and finalize its output.
    /// Metadata/eligibility errors are retryable and do not acquire input or write.
    /// After encoding starts, any error or unwind permanently poisons the writer.
    /// `is_last` must be supplied explicitly; finish never invents a final frame.
    pub fn write_frame<S: RegionSource + ?Sized>(
        &mut self,
        source: &mut S,
        frame: &FrameOptions,
        options: &EncodeOptions,
        is_last: bool,
    ) -> Result<FrameProgress> {
        self.write_frame_impl(source, frame, options, is_last, false)
    }

    /// Select this displayed full-canvas REPLACE frame as a jxli keyframe.
    /// Requires config.index and must be used for the first frame when indexing.
    /// Later interval frames cannot depend on references from earlier intervals.
    pub fn write_indexed_frame<S: RegionSource + ?Sized>(
        &mut self,
        source: &mut S,
        frame: &FrameOptions,
        options: &EncodeOptions,
        is_last: bool,
    ) -> Result<FrameProgress> {
        self.write_frame_impl(source, frame, options, is_last, true)
    }

    fn write_frame_impl<S: RegionSource + ?Sized>(
        &mut self,
        source: &mut S,
        frame: &FrameOptions,
        options: &EncodeOptions,
        is_last: bool,
        selected: bool,
    ) -> Result<FrameProgress> {
        self.check_open()?;
        if selected && self.index.is_none() {
            return Err(Error::InvalidInput("frame indexing was not enabled".into()));
        }
        if options.threads != self.config.threads {
            return Err(Error::InvalidInput(
                "frame threads differ from stream configuration".into(),
            ));
        }
        if self.next_frame >= 0x8000_0000 {
            return Err(Error::Limit("jxlp chunk count"));
        }
        self.poisoned = true;
        let descriptor = source.descriptor().clone();
        self.poisoned = false;
        descriptor.validate()?;
        crate::encode::validate_geometry(self.canvas.geometry(), options)?;
        crate::encode::validate_geometry(descriptor.geometry(), options)?;
        let mut metadata = descriptor.metadata().clone();
        // Animation timing is a canvas contract, not input packing metadata.
        metadata.animation = self.canvas.metadata().animation;
        if metadata != *self.canvas.metadata()
            || descriptor.color_channels() != self.canvas.color_channels()
        {
            return Err(Error::InvalidInput(
                "frame metadata differs from canvas".into(),
            ));
        }
        let mut references = crate::encode::validate_frame_references(
            self.canvas.geometry(),
            descriptor.geometry(),
            frame,
            options,
            is_last,
            self.references,
        )?;
        let index_update = self
            .index
            .as_mut()
            .map(|index| {
                index.prepare(
                    self.canvas.geometry(),
                    descriptor.geometry(),
                    frame,
                    options,
                    is_last,
                    selected,
                    self.codestream_bytes,
                )
            })
            .transpose()?;
        let reserve_index = index_update.as_ref().map_or(Ok(0), |update| {
            update
                .payload_bytes
                .checked_add(16)
                .ok_or(Error::Limit("frame index box bytes"))
        })?;
        let initial = if self.started {
            0
        } else {
            crate::container::PREFIX.len() + 9
        } + if self.next_frame == 0 {
            self.image_header.len()
        } else {
            0
        };
        let remaining = self
            .config
            .max_output_bytes
            .checked_sub(self.writer.end())
            .and_then(|n| n.checked_sub(initial))
            .and_then(|n| n.checked_sub(20))
            .and_then(|n| n.checked_sub(reserve_index))
            .filter(|&n| n > 0)
            .ok_or(Error::Limit("streaming output bytes"))?;
        let mut bounded = options.clone();
        bounded.max_output_bytes = bounded.max_output_bytes.min(remaining);
        let decision = groupwise_decision_geometry(
            descriptor.width(),
            descriptor.height(),
            &bounded.modular,
            bounded.group_size_shift,
            &bounded.passes,
            bounded.responsive,
            bounded.patches.is_some() || bounded.source_patches.is_some(),
        )?;
        let area = if decision == GroupwiseDecision::FrameAreas {
            Some(AreaFrame::new(
                FrameAreaPlan::new(descriptor.clone(), &bounded)?,
                self.canvas.geometry(),
                frame,
                &bounded,
                is_last,
            )?)
        } else {
            match self.config.buffering {
                FrameBuffering::StrictAreas => {
                    return Err(Error::InvalidInput(format!(
                        "region encoding requires source frame-area eligibility: {decision:?}",
                    )));
                }
                FrameBuffering::CurrentFrame { max_sample_bytes } => {
                    if region_input::buffered_sample_bytes(&descriptor)? > max_sample_bytes {
                        return Err(Error::Limit("buffered frame sample bytes"));
                    }
                }
            }
            None
        };
        // Set before any user callback or codec work. This deliberately remains
        // true if a provider, lease release, codec task or sink unwinds.
        self.poisoned = true;
        let mode = if area.is_some() {
            FrameProgressMode::FrameAreas
        } else {
            FrameProgressMode::CurrentFrameBuffered
        };
        let buffered = if area.is_none() {
            let FrameBuffering::CurrentFrame { max_sample_bytes } = self.config.buffering else {
                unreachable!()
            };
            let image = region_input::read_frame(&descriptor, source, max_sample_bytes)?;
            let frame = Frame {
                image: &image,
                options: frame.clone(),
            };
            let canvas = self.canvas.geometry();
            let encode =
                || crate::encode::encode_logical_frame_geometry(canvas, &frame, &bounded, is_last);
            Some(match &self.pool {
                Some(pool) => execution::with_pool(pool, encode),
                None => execution::serial(encode),
            }?)
        } else {
            None
        };
        self.start_output()?;
        let box_start = self.writer.position();
        self.writer.write(b"\0\0\0\x01jxlp\0\0\0\0\0\0\0\x14")?;
        self.writer
            .write(&(self.next_frame | if is_last { 0x8000_0000 } else { 0 }).to_be_bytes())?;
        if self.next_frame == 0 {
            self.writer.write(&self.image_header)?;
        }
        let frame_start = self.writer.position();
        if let Some(area) = area {
            area.emit(source, self.pool.as_ref(), &mut self.writer)?;
        } else if let Some((bytes, generated_reference)) = buffered {
            self.writer.write(&bytes)?;
            if generated_reference {
                references[3] = Some(true);
            }
        }
        let size = u64::try_from(self.writer.end() - box_start)
            .map_err(|_| Error::Limit("streaming box size"))?;
        self.writer.seek(box_start + 8)?;
        self.writer.write(&size.to_be_bytes())?;
        self.writer.finalize()?;
        self.codestream_bytes = self
            .codestream_bytes
            .checked_add((self.writer.end() - frame_start) as u64)
            .ok_or(Error::Limit("streaming codestream bytes"))?;
        if let Some(update) = index_update {
            let index = self.index.as_mut().expect("validated index configuration");
            index.commit(update);
            if is_last {
                let payload = index.payload()?;
                self.write_box(*b"jxli", &payload)?;
                self.writer.finalize()?;
                self.index_written = true;
            }
        }
        let progress = FrameProgress {
            frame_index: self.next_frame,
            mode,
            bytes_written: self.writer.end(),
            finalized_bytes: self.writer.finalized(),
            is_last,
        };
        self.references = references;
        self.next_frame += 1;
        self.complete = is_last;
        self.poisoned = false;
        self.image_header.clear();
        Ok(progress)
    }

    /// Immutable prefix length relative to the sink's initial position. A later
    /// failed frame may have written beyond this boundary, never before it.
    pub fn finalized_bytes(&self) -> usize {
        self.writer.finalized()
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Write one complete nonstructural box now, without retaining its payload.
    /// Supports the existing Exif/xml /jumb/jxli allowlist. Raw jxli is forbidden
    /// when automatic indexing is enabled, and duplicates/structural boxes fail.
    /// May precede, separate or follow frames; this explicit call can write the
    /// container prefix before the first frame's eligibility is known.
    pub fn write_metadata(&mut self, kind: [u8; 4], payload: &[u8]) -> Result<usize> {
        if self.poisoned {
            return Err(Error::InvalidInput("stream is poisoned".into()));
        }
        if !matches!(&kind, b"Exif" | b"xml " | b"jumb" | b"jxli")
            || (kind == *b"Exif" && payload.len() < 4)
            || (kind == *b"jxli" && (self.index.is_some() || self.index_written))
        {
            return Err(Error::InvalidInput(
                "invalid metadata box type or payload".into(),
            ));
        }
        let prefix = if self.started {
            0
        } else {
            crate::container::PREFIX.len() + 9
        };
        let reserved = if self.complete {
            0
        } else {
            self.index
                .as_ref()
                .map_or(Ok(0), IndexState::reserved_bytes)?
        };
        self.writer
            .end()
            .checked_add(prefix)
            .and_then(|n| n.checked_add(16))
            .and_then(|n| n.checked_add(payload.len()))
            .and_then(|n| n.checked_add(reserved))
            .filter(|&n| n <= self.config.max_output_bytes)
            .ok_or(Error::Limit("streaming output bytes"))?;
        self.poisoned = true;
        self.start_output()?;
        self.write_box(kind, payload)?;
        self.writer.finalize()?;
        self.index_written |= kind == *b"jxli";
        self.poisoned = false;
        Ok(self.writer.finalized())
    }

    /// Requires a successfully finalized last frame. Returns the sink and total
    /// encoded length, without flushing, truncating or closing the sink.
    pub fn finish(self) -> Result<(W, usize)> {
        if self.poisoned || !self.complete {
            return Err(Error::InvalidInput(
                "stream is poisoned or has no final frame".into(),
            ));
        }
        self.writer.finish_into()
    }

    fn check_open(&self) -> Result<()> {
        if self.poisoned || self.complete {
            return Err(Error::InvalidInput(
                "stream is poisoned or already complete".into(),
            ));
        }
        Ok(())
    }

    fn start_output(&mut self) -> Result<()> {
        if !self.started {
            self.writer.write(crate::container::PREFIX)?;
            self.writer.write(b"\0\0\0\x09jxll\x0a")?;
            self.started = true;
        }
        Ok(())
    }

    fn write_box(&mut self, kind: [u8; 4], payload: &[u8]) -> Result<()> {
        let size = u64::try_from(payload.len())
            .ok()
            .and_then(|n| n.checked_add(16))
            .ok_or(Error::Limit("streaming box size"))?;
        self.writer.write(&1u32.to_be_bytes())?;
        self.writer.write(&kind)?;
        self.writer.write(&size.to_be_bytes())?;
        self.writer.write(payload)
    }
}

#[cfg(test)]
mod tests;
