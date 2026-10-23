// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! FrameInfo-aware lossless search, matching the frame callback contract of
//! enc_lossless_search.cc at b866b3d. Trials retain real frame bytes, never
//! independent single-image codestreams. Caller-provided blend/reference
//! semantics remain unchanged by parameter selection.

use super::{
    CandidateParams, DriverOptions,
    driver::SearchReport,
    native::{
        FrameContext, NativeAdaptation, NativeArtifact, NativeBackendOptions,
        run_native_search_with_pool,
    },
};
use crate::{EncodeOptions, Error, Image, Result, frame::Frame};

#[derive(Debug)]
pub struct FrameSearchReport {
    pub params: CandidateParams,
    pub artifact: NativeArtifact,
    pub report: SearchReport,
    pub adaptations: Vec<NativeAdaptation>,
    /// Includes a private patch-reference frame when that candidate uses one.
    pub encoded_bytes: usize,
}

#[derive(Debug)]
pub struct SequenceSearchResult {
    pub bytes: Vec<u8>,
    pub frames: Vec<FrameSearchReport>,
}

/// Search each supplied frame, reusing one worker pool and assembling one
/// image header and container. `full_limits` bound the complete sequence's
/// aggregate full-frame work and total output, not each frame separately.
/// Probe limits apply separately to each frame. No implicit difference or
/// alpha processing is performed; use `multiframe::plan` for exact crops.
pub fn run_native_sequence_search(
    canvas: &Image,
    frames: &[Frame<'_>],
    params: &CandidateParams,
    driver: &DriverOptions,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
) -> Result<SequenceSearchResult> {
    crate::encode::validate_frames(canvas, frames, base)?;
    params.validate()?;
    let maximum = base
        .max_output_bytes
        .min(driver.full_limits.max_output_bytes);
    let mut stream = crate::codestream::image_header(canvas, maximum)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(base.threads)
        .build()
        .map_err(|e| Error::Encoding(e.to_string()))?;
    let mut limits = driver.full_limits;
    let mut reports = Vec::new();
    reports
        .try_reserve_exact(frames.len())
        .map_err(|_| Error::Limit("frame reports"))?;
    for (index, frame) in frames.iter().enumerate() {
        let mut frame_driver = driver.clone();
        frame_driver.full_limits = limits;
        frame_driver.full_limits.max_output_bytes = maximum
            .checked_sub(stream.len())
            .filter(|&n| n > 0)
            .ok_or(Error::Limit("sequence output bytes"))?;
        let result = run_native_search_with_pool(
            frame.image,
            params,
            &frame_driver,
            base,
            settings,
            &pool,
            Some(FrameContext {
                canvas,
                options: &frame.options,
                last: index + 1 == frames.len(),
            }),
        )?;
        let output = result.output;
        limits.max_full_encodes = limits
            .max_full_encodes
            .checked_sub(output.report.full_encodes)
            .ok_or(Error::Limit("sequence full-frame work"))?;
        limits.max_full_pixels = limits
            .max_full_pixels
            .checked_sub(output.report.full_pixels)
            .ok_or(Error::Limit("sequence full-frame work"))?;
        let encoded_bytes = output.bytes.len();
        stream
            .try_reserve(encoded_bytes)
            .map_err(|_| Error::Limit("sequence output allocation"))?;
        stream.extend(output.bytes);
        reports.push(FrameSearchReport {
            params: output.params,
            artifact: output.artifact,
            report: output.report,
            adaptations: result.adaptations,
            encoded_bytes,
        });
    }
    Ok(SequenceSearchResult {
        bytes: crate::encode::wrap_stream(canvas, stream, maximum)?,
        frames: reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout, frame::BlendInfo};

    fn image() -> Image {
        Image::new(8, 8, PixelLayout::Gray, PixelData::U8(vec![7; 64])).unwrap()
    }

    #[test]
    fn layer_search_preserves_context_and_records_each_actual_winner() {
        let canvas = image();
        let patch = crate::multiframe::crop(
            &canvas,
            crate::multiframe::Rect {
                x: 1,
                y: 2,
                width: 2,
                height: 3,
            },
        )
        .unwrap();
        let mut first = Frame::new(&canvas);
        first.options.save_as_reference = 1;
        let mut next = Frame::new(&patch);
        next.options.origin = (1, 2);
        next.options.blend = BlendInfo {
            source: 1,
            ..Default::default()
        };
        let result = run_native_sequence_search(
            &canvas,
            &[first, next],
            &CandidateParams {
                speed_tier: 9,
                ..Default::default()
            },
            &DriverOptions::default(),
            &EncodeOptions::default(),
            NativeBackendOptions::default(),
        )
        .unwrap();
        assert_eq!(result.frames.len(), 2);
        for frame in &result.frames {
            assert_eq!(
                frame.report.trials[frame.report.winner_index].bytes,
                Some(frame.encoded_bytes)
            );
            assert!(frame.adaptations.iter().all(|a| a.gaps.is_empty()));
        }
        assert!(result.bytes.len() > result.frames.iter().map(|f| f.encoded_bytes).sum::<usize>());
    }

    #[test]
    fn full_work_and_output_limits_are_sequence_wide() {
        let canvas = image();
        let frames = [Frame::new(&canvas), Frame::new(&canvas)];
        let params = CandidateParams {
            speed_tier: 9,
            ..Default::default()
        };
        let mut options = DriverOptions::default();
        options.full_limits.max_full_encodes = 1;
        assert!(matches!(
            run_native_sequence_search(
                &canvas,
                &frames,
                &params,
                &options,
                &EncodeOptions::default(),
                NativeBackendOptions::default()
            ),
            Err(Error::Limit(_))
        ));
        options.full_limits.max_full_encodes = 100;
        options.full_limits.max_full_pixels = 64;
        assert!(matches!(
            run_native_sequence_search(
                &canvas,
                &frames,
                &params,
                &options,
                &EncodeOptions::default(),
                NativeBackendOptions::default()
            ),
            Err(Error::Limit(_))
        ));
        options.full_limits.max_full_pixels = 100_000;
        options.full_limits.max_output_bytes = 1;
        assert!(
            run_native_sequence_search(
                &canvas,
                &frames,
                &params,
                &options,
                &EncodeOptions::default(),
                NativeBackendOptions::default()
            )
            .is_err()
        );
    }
}
