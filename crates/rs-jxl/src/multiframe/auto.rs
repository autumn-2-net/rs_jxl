//! Per-logical-frame search using real encoded layers, with retained winners.
use super::{Rect, Reference, compatible, crop, difference_bounds, regions};
use crate::{
    EncodeOptions, Error, Image, Result,
    frame::{BlendInfo, BlendMode, Frame, FrameOptions},
};
use std::{
    borrow::Cow,
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
pub struct AutoOptions {
    /// SmallestCrop tests both first/previous references using actual bytes.
    pub reference: Reference,
    pub allow_mask: bool,
    pub allow_multi_rect: bool,
    pub tile_size: u32,
    pub max_rectangles: usize,
    /// Logical input frames, independent of inserted zero-duration layers.
    pub max_frames: usize,
    pub max_total_trials: usize,
    /// Sum of pixel areas actually submitted to the encoder across all trials.
    pub max_trial_pixels: u64,
}
impl Default for AutoOptions {
    fn default() -> Self {
        Self {
            reference: Reference::SmallestCrop,
            allow_mask: true,
            allow_multi_rect: true,
            tile_size: 32,
            max_rectangles: 32,
            max_frames: usize::MAX,
            max_total_trials: usize::MAX,
            max_trial_pixels: u64::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Full,
    CropPrevious,
    CropFirst,
    MaskPrevious,
    MaskFirst,
    TilesPrevious,
    TilesFirst,
}

#[derive(Debug)]
pub struct AutoCandidateReport {
    pub strategy: Strategy,
    /// Actual layer bytes, including frame headers and any private patch atlas.
    pub encoded_bytes: usize,
    /// Regular layers, excluding private reference-only patch frames.
    pub encoded_frames: usize,
    pub elapsed: Duration,
}
#[derive(Debug)]
pub struct AutoFrameReport {
    pub winner_index: usize,
    pub candidates: Vec<AutoCandidateReport>,
}
#[derive(Debug)]
pub struct AutoEncodeResult {
    pub bytes: Vec<u8>,
    /// One report per input, also when its duration is zero.
    pub frames: Vec<AutoFrameReport>,
}

struct Candidate {
    strategy: Strategy,
    reference: Option<usize>,
    source: u8,
    rects: Vec<Rect>,
    mask: bool,
}

/// Encode exact full-image states. Candidates share metadata, final reference
/// contents and timing, so retained frame bytes can be assembled without re-encoding.
/// Selection minimizes bytes, with stable ties preferring full/crop over masking
/// and splitting. This is a local frame minimum, not a global compression proof.
pub fn encode_auto(
    images: &[Image],
    durations: &[u32],
    options: &EncodeOptions,
    auto: &AutoOptions,
) -> Result<AutoEncodeResult> {
    if images.is_empty() || images.len() != durations.len() {
        return Err(Error::InvalidInput(
            "nonempty images and one duration per frame are required".into(),
        ));
    }
    if images.len() > auto.max_frames {
        return Err(Error::Limit("automatic logical frame count"));
    }
    if auto.tile_size == 0 || auto.max_rectangles == 0 {
        return Err(Error::InvalidInput(
            "tile_size and max_rectangles must be positive".into(),
        ));
    }
    let canvas = &images[0];
    for image in images {
        compatible(canvas, image)?;
        crate::encode::validate_geometry(image.into(), options)?;
    }
    if canvas.metadata().animation.is_none() && durations.iter().any(|&d| d != 0) {
        return Err(Error::InvalidInput(
            "nonzero durations require animation metadata".into(),
        ));
    }
    let work = || encode_inner(images, durations, options, auto);
    if options.threads == 1 || crate::execution::is_serial() {
        return crate::execution::serial(work);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .build()
        .map_err(|e| Error::Encoding(e.to_string()))?;
    crate::execution::with_pool(&pool, work)
}

fn candidates(images: &[Image], index: usize, auto: &AutoOptions) -> Result<Vec<Candidate>> {
    let current = &images[index];
    let full = Rect {
        x: 0,
        y: 0,
        width: current.width(),
        height: current.height(),
    };
    let mut result = vec![Candidate {
        strategy: Strategy::Full,
        reference: None,
        source: 0,
        rects: vec![full],
        mask: false,
    }];
    if index == 0
        || auto.reference == Reference::Full
        || current
            .metadata()
            .extra_channels
            .iter()
            .any(|e| e.dim_shift != 0)
    {
        return Ok(result);
    }
    let references = match auto.reference {
        Reference::Previous => vec![index - 1],
        Reference::First => vec![0],
        _ if index == 1 => vec![0],
        _ => vec![index - 1, 0],
    };
    for reference in references {
        let previous = &images[reference];
        let first = reference == 0 && (index != 1 || auto.reference == Reference::First);
        let source = if reference == 0 { 1 } else { 2 };
        let bounds = difference_bounds(previous, current)?;
        let rect = bounds.unwrap_or(Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        });
        // A full-size REPLACE crop is the full candidate already tested.
        if rect != full {
            result.push(Candidate {
                strategy: if first {
                    Strategy::CropFirst
                } else {
                    Strategy::CropPrevious
                },
                reference: Some(reference),
                source,
                rects: vec![rect],
                mask: false,
            });
        }
        if bounds.is_none() {
            continue;
        }
        if auto.allow_mask && regions::mask_eligible(current) && regions::mask_eligible(previous) {
            result.push(Candidate {
                strategy: if first {
                    Strategy::MaskFirst
                } else {
                    Strategy::MaskPrevious
                },
                reference: Some(reference),
                source,
                rects: vec![rect],
                mask: true,
            });
        }
        if auto.allow_multi_rect
            && let Some(rects) =
                regions::tiles(previous, current, auto.tile_size, auto.max_rectangles)
            && rects.len() > 1
        {
            result.push(Candidate {
                strategy: if first {
                    Strategy::TilesFirst
                } else {
                    Strategy::TilesPrevious
                },
                reference: Some(reference),
                source,
                rects,
                mask: false,
            });
        }
    }
    Ok(result)
}

fn encode_inner(
    images: &[Image],
    durations: &[u32],
    options: &EncodeOptions,
    auto: &AutoOptions,
) -> Result<AutoEncodeResult> {
    let canvas = &images[0];
    let mut stream = crate::codestream::image_header(canvas, options.max_output_bytes)?;
    let mut reports = Vec::new();
    reports
        .try_reserve_exact(images.len())
        .map_err(|_| Error::Limit("automatic frame reports"))?;
    let mut available = [None; 4];
    let mut trials = 0usize;
    let mut pixels = 0u64;
    for (index, image) in images.iter().enumerate() {
        let mut report = AutoFrameReport {
            winner_index: 0,
            candidates: Vec::new(),
        };
        let mut winner: Option<Vec<u8>> = None;
        let mut winner_references = available;
        for candidate in candidates(images, index, auto)? {
            trials = trials
                .checked_add(1)
                .filter(|&n| n <= auto.max_total_trials)
                .ok_or(Error::Limit("automatic trial count"))?;
            let mut trial_refs = available;
            let started = Instant::now();
            let mut bytes = Vec::new();
            for (part, &rect) in candidate.rects.iter().enumerate() {
                pixels = pixels
                    .checked_add(u64::from(rect.width) * u64::from(rect.height))
                    .filter(|&n| n <= auto.max_trial_pixels)
                    .ok_or(Error::Limit("automatic trial pixel work"))?;
                let final_part = part + 1 == candidate.rects.len();
                let last = index + 1 == images.len() && final_part;
                let frame_image = if candidate.mask {
                    Cow::Owned(regions::mask(
                        &images[candidate.reference.expect("mask reference")],
                        image,
                        rect,
                    )?)
                } else if rect.x == 0
                    && rect.y == 0
                    && rect.width == image.width()
                    && rect.height == image.height()
                {
                    Cow::Borrowed(image)
                } else {
                    Cow::Owned(crop(image, rect)?)
                };
                let frame = Frame {
                    image: &frame_image,
                    options: FrameOptions {
                        origin: (rect.x as i32, rect.y as i32),
                        duration: if final_part { durations[index] } else { 0 },
                        save_as_reference: if last {
                            0
                        } else if index == 0 {
                            1
                        } else {
                            2
                        },
                        blend: BlendInfo {
                            source: if part == 0 { candidate.source } else { 2 },
                            mode: if candidate.mask {
                                BlendMode::Blend
                            } else {
                                BlendMode::Replace
                            },
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                };
                trial_refs = crate::encode::validate_frame_references(
                    canvas.into(),
                    frame.image.into(),
                    &frame.options,
                    options,
                    last,
                    trial_refs,
                )?;
                let encoded =
                    crate::encode::encode_logical_frame_trial(canvas, &frame, options, last, None)?
                        .into_complete()?;
                append(&mut bytes, encoded, options.max_output_bytes)?;
            }
            report.candidates.push(AutoCandidateReport {
                strategy: candidate.strategy,
                encoded_bytes: bytes.len(),
                encoded_frames: candidate.rects.len(),
                elapsed: started.elapsed(),
            });
            if winner.as_ref().is_none_or(|best| bytes.len() < best.len()) {
                report.winner_index = report.candidates.len() - 1;
                winner_references = trial_refs;
                winner = Some(bytes);
            }
        }
        append(
            &mut stream,
            winner.expect("full candidate always exists"),
            options.max_output_bytes,
        )?;
        available = winner_references;
        reports.push(report);
    }
    Ok(AutoEncodeResult {
        bytes: crate::encode::wrap_stream(canvas, stream, options.max_output_bytes)?,
        frames: reports,
    })
}

fn append(target: &mut Vec<u8>, bytes: Vec<u8>, maximum: usize) -> Result<()> {
    target
        .len()
        .checked_add(bytes.len())
        .filter(|&n| n <= maximum)
        .ok_or(Error::Limit("encoded output bytes"))?;
    target
        .try_reserve(bytes.len())
        .map_err(|_| Error::Limit("automatic output allocation"))?;
    target.extend(bytes);
    Ok(())
}
