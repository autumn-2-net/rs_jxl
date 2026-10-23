//! Incremental output: frame-buffered Write, or one-pass region input with
//! Write+Seek and source-compatible within-frame directory backpatching.

use crate::{EncodeOptions, Error, Image, Result, container::ChunkedWriter, frame::Frame};
use std::io::Write;

mod index;
mod seek_output;
mod seekable;
mod sequence;
mod toc;

pub use seekable::encode_regions_seekable_to;
pub use sequence::{
    FrameBuffering, FrameEncoder, FrameEncoderOptions, FrameIndexOptions, FrameProgress,
    FrameProgressMode,
};

/// Validate the sequence before writing, then encode and emit one logical
/// frame at a time. Codec/I/O errors after the header may leave partial output;
/// use `encode_frames_to` when the sink must stay untouched on codec failure.
pub fn encode_frames_streaming_to(
    canvas: &Image,
    frames: &[Frame<'_>],
    options: &EncodeOptions,
    output: impl Write,
) -> Result<()> {
    crate::encode::validate_frames(canvas, frames, options)?;
    encode_validated_to(canvas, frames, options, output, None)
}

/// Emit an ordered jxlp container and one jxli index. Selections refer to the
/// input slice, start at zero and identify independently displayed full-frame
/// replacements. Later frames may use only references saved in their interval.
/// The timebase and durations follow the animation, not frozen source bugs.
pub fn encode_indexed_frames_streaming_to(
    canvas: &Image,
    frames: &[Frame<'_>],
    selected_frames: &[usize],
    options: &EncodeOptions,
    output: impl Write,
) -> Result<()> {
    crate::encode::validate_frames(canvas, frames, options)?;
    if options.patches.is_some() || options.source_patches.is_some() {
        return Err(Error::InvalidInput(
            "indexed output does not support generated patch reference frames".into(),
        ));
    }
    let index = index::IndexPlan::new(canvas, frames, selected_frames)?;
    encode_validated_to(canvas, frames, options, output, Some(index))
}

fn encode_validated_to(
    canvas: &Image,
    frames: &[Frame<'_>],
    options: &EncodeOptions,
    output: impl Write,
    mut index: Option<index::IndexPlan>,
) -> Result<()> {
    let header = crate::execution::serial(|| {
        crate::codestream::image_header(canvas, options.max_output_bytes)
    })?;
    let mut codestream_offset = header.len() as u64;
    let level = if crate::codestream::container_prefix(canvas, 0)?.is_empty() {
        5
    } else {
        10
    };
    let pool = if options.threads == 1 || crate::execution::is_serial() {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(options.threads)
                .build()
                .map_err(|e| Error::Encoding(e.to_string()))?,
        )
    };
    let mut writer = ChunkedWriter::new(output, level, options.max_output_bytes)?;
    writer.write_chunk(&header, false)?;
    for (i, frame) in frames.iter().enumerate() {
        if let Some(index) = &mut index {
            index.record(i, codestream_offset)?;
        }
        let mut bounded = options.clone();
        // Twelve bytes is the minimum jxlp wrapper; the writer checks the
        // extended-size case and complete remaining limit before mutation.
        bounded.max_output_bytes = options
            .max_output_bytes
            .checked_sub(writer.bytes_written())
            .and_then(|n| n.checked_sub(12))
            .filter(|&n| n > 0)
            .ok_or(Error::Limit("streaming output bytes"))?;
        let last = i + 1 == frames.len();
        let encode =
            || crate::encode::encode_logical_frame_trial(canvas, frame, &bounded, last, None);
        let bytes = match &pool {
            Some(pool) => crate::execution::with_pool(pool, encode),
            None => crate::execution::serial(encode),
        }?
        .into_complete()?;
        writer.write_chunk(&bytes, last)?;
        codestream_offset = codestream_offset
            .checked_add(bytes.len() as u64)
            .ok_or(Error::Limit("streaming codestream offset"))?;
    }
    if let Some(index) = index {
        let available = options
            .max_output_bytes
            .checked_sub(writer.bytes_written())
            .and_then(|n| n.checked_sub(8))
            .ok_or(Error::Limit("frame index box bytes"))?;
        let payload = index.payload(available)?;
        writer.write_metadata(crate::container::frame_index::BOX_TYPE, &payload)?;
    }
    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout};

    #[test]
    fn prevalidation_leaves_sink_untouched_and_output_budget_is_total() {
        let image = Image::new(8, 8, PixelLayout::Gray, PixelData::U8(vec![3; 64])).unwrap();
        let mut bytes = Vec::new();
        assert!(
            encode_frames_streaming_to(&image, &[], &EncodeOptions::default(), &mut bytes).is_err()
        );
        assert!(bytes.is_empty());
        let frames = [Frame::new(&image), Frame::new(&image)];
        encode_frames_streaming_to(&image, &frames, &EncodeOptions::default(), &mut bytes).unwrap();
        let expected = bytes.len();
        for maximum in [expected, expected - 1] {
            let mut out = Vec::new();
            let result = encode_frames_streaming_to(
                &image,
                &frames,
                &EncodeOptions {
                    threads: 2,
                    max_output_bytes: maximum,
                    ..Default::default()
                },
                &mut out,
            );
            assert_eq!(result.is_ok(), maximum == expected);
            assert!(out.len() <= maximum);
        }
    }

    #[test]
    fn sink_remains_on_calling_thread_with_parallel_frame_encoding() {
        use std::{cell::RefCell, rc::Rc};
        struct LocalSink(Rc<RefCell<Vec<u8>>>);
        impl Write for LocalSink {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.borrow_mut().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let output = Rc::new(RefCell::new(Vec::new()));
        let image = Image::new(8, 8, PixelLayout::Gray, PixelData::U8(vec![3; 64])).unwrap();
        encode_frames_streaming_to(
            &image,
            &[Frame::new(&image)],
            &EncodeOptions {
                threads: 2,
                ..Default::default()
            },
            LocalSink(output.clone()),
        )
        .unwrap();
        assert!(!output.borrow().is_empty());
    }

    #[test]
    fn indexed_prevalidation_and_total_box_budget_are_enforced() {
        let image = Image::new(8, 8, PixelLayout::Gray, PixelData::U8(vec![3; 64])).unwrap();
        let mut metadata = image.metadata().clone();
        metadata.animation = Some(crate::metadata::Animation::default());
        let image = image.with_metadata(metadata, vec![]).unwrap();
        let frames = [
            Frame {
                image: &image,
                options: crate::frame::FrameOptions {
                    duration: 5,
                    ..Default::default()
                },
            },
            Frame::new(&image),
        ];
        let options = EncodeOptions::default();
        for selection in [&[][..], &[1], &[0, 0], &[0, 2]] {
            let mut out = Vec::new();
            assert!(
                encode_indexed_frames_streaming_to(&image, &frames, selection, &options, &mut out)
                    .is_err()
            );
            assert!(out.is_empty());
        }
        let mut bytes = Vec::new();
        let patches = EncodeOptions {
            patches: Some(Default::default()),
            ..Default::default()
        };
        assert!(
            encode_indexed_frames_streaming_to(&image, &frames, &[0, 1], &patches, &mut bytes)
                .is_err()
        );
        assert!(bytes.is_empty());
        encode_indexed_frames_streaming_to(&image, &frames, &[0, 1], &options, &mut bytes).unwrap();
        for maximum in [bytes.len(), bytes.len() - 1] {
            let options = EncodeOptions {
                max_output_bytes: maximum,
                ..Default::default()
            };
            let mut out = Vec::new();
            let result =
                encode_indexed_frames_streaming_to(&image, &frames, &[0, 1], &options, &mut out);
            assert_eq!(result.is_ok(), maximum == bytes.len());
            assert!(out.len() <= maximum);
        }
    }
}
