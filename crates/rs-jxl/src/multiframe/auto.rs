//! Per-logical-frame search using real encoded layers, with retained winners.
use super::{Rect, Reference, compatible, crop, diff::DiffMap, partition, regions};
use crate::{
    EncodeOptions, Error, Image, Result,
    encode::budget::{SectionBudget, TrialEncode},
    frame::{BlendInfo, BlendMode, Frame, FrameOptions},
};
use std::{
    borrow::Cow,
    sync::Arc,
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
    /// Stop only when finalized output proves a trial cannot beat the winner.
    pub prune: bool,
    /// Add a cost-estimated rectangle proposal; retain all legacy candidates.
    pub adaptive_rectangles: bool,
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
            prune: true,
            adaptive_rectangles: true,
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
    MergedPrevious,
    MergedFirst,
}

#[derive(Debug)]
pub struct AutoCandidateReport {
    pub strategy: Strategy,
    /// Actual layer bytes, including frame headers and any private patch atlas.
    pub encoded_bytes: usize,
    /// Regular layers, excluding private reference-only patch frames.
    pub encoded_frames: usize,
    /// Pixel area actually submitted to the encoder, excluding analysis scans.
    pub trial_pixels: u64,
    pub elapsed: Duration,
}
#[derive(Debug)]
pub struct AutoPrunedReport {
    pub strategy: Strategy,
    /// Proven size lower bound, not an estimated or complete encoded byte count.
    pub lower_bound: usize,
    pub encoded_frames: usize,
    pub planned_frames: usize,
    pub trial_pixels: u64,
    pub elapsed: Duration,
}
#[derive(Debug)]
pub struct AutoDiffReport {
    pub reference: usize,
    pub changed_pixels: u64,
    pub bounds: Option<Rect>,
}
#[derive(Debug)]
pub struct AutoFrameReport {
    pub winner_index: usize,
    pub candidates: Vec<AutoCandidateReport>,
    pub pruned: Vec<AutoPrunedReport>,
    pub differences: Vec<AutoDiffReport>,
    pub analysis_elapsed: Duration,
}
#[derive(Debug)]
pub struct AutoEncodeResult {
    pub bytes: Vec<u8>,
    /// One report per input, also when its duration is zero.
    pub frames: Vec<AutoFrameReport>,
}

pub(super) struct Candidate {
    pub(super) strategy: Strategy,
    diff: Option<Arc<DiffMap>>,
    pub(super) source: u8,
    pub(super) rects: Vec<Rect>,
    mask: bool,
}

impl Candidate {
    // Ranking only. No candidate is discarded using this estimate.
    fn estimated_work(&self) -> u64 {
        let area = self.rects.iter().fold(0u64, |sum, rect| {
            sum.saturating_add(u64::from(rect.width) * u64::from(rect.height))
        });
        let pixels = self.diff.as_ref().map_or(area, |diff| {
            diff.changed_pixels().saturating_add(area.div_ceil(64))
        });
        pixels.saturating_add((self.rects.len() as u64).saturating_mul(64))
    }
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
    validate(images, durations, options, auto)?;
    if options.threads == 1 || crate::execution::is_serial() {
        return encode_auto_in_pool(images, durations, options, auto, None);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .build()
        .map_err(|e| Error::Encoding(e.to_string()))?;
    encode_auto_in_pool(images, durations, options, auto, Some(&pool))
}

pub(crate) fn encode_auto_in_pool(
    images: &[Image],
    durations: &[u32],
    options: &EncodeOptions,
    auto: &AutoOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<AutoEncodeResult> {
    validate(images, durations, options, auto)?;
    let work = || encode_inner(images, durations, options, auto);
    match pool {
        Some(pool) => crate::execution::with_pool(pool, work),
        None => crate::execution::serial(work),
    }
}

fn validate(
    images: &[Image],
    durations: &[u32],
    options: &EncodeOptions,
    auto: &AutoOptions,
) -> Result<()> {
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
    Ok(())
}

fn candidates(
    images: &[Image],
    index: usize,
    auto: &AutoOptions,
    bytes_per_pixel: f64,
    differences: &mut Vec<AutoDiffReport>,
) -> Result<Vec<Candidate>> {
    let mut maps = Vec::new();
    if index != 0
        && auto.reference != Reference::Full
        && !images[index]
            .metadata()
            .extra_channels
            .iter()
            .any(|e| e.dim_shift != 0)
    {
        for reference in reference_indices(index, auto.reference) {
            let diff = Arc::new(DiffMap::between(&images[reference], &images[index])?);
            differences.push(AutoDiffReport {
                reference,
                changed_pixels: diff.changed_pixels(),
                bounds: diff.bounds(),
            });
            maps.push((reference, diff));
        }
    }
    candidates_from_maps(&images[index], index, auto, bytes_per_pixel, &maps)
}

pub(super) fn reference_indices(index: usize, reference: Reference) -> Vec<usize> {
    if index == 0 || reference == Reference::Full {
        return Vec::new();
    }
    match reference {
        Reference::Previous => vec![index - 1],
        Reference::First => vec![0],
        _ if index == 1 => vec![0],
        _ => vec![index - 1, 0],
    }
}

/// Shared geometry proposal stage. Maps are computed once per reference pair;
/// analysis never materializes cropped images or starts an encode trial.
pub(super) fn candidates_from_maps(
    current: &Image,
    index: usize,
    auto: &AutoOptions,
    bytes_per_pixel: f64,
    maps: &[(usize, Arc<DiffMap>)],
) -> Result<Vec<Candidate>> {
    let full = Rect {
        x: 0,
        y: 0,
        width: current.width(),
        height: current.height(),
    };
    let mut result = vec![Candidate {
        strategy: Strategy::Full,
        diff: None,
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
    let mut merged = Vec::new();
    for &(reference, ref diff) in maps {
        let first = reference == 0 && (index != 1 || auto.reference == Reference::First);
        let source = if reference == 0 { 1 } else { 2 };
        let bounds = diff.bounds();
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
                diff: None,
                source,
                rects: vec![rect],
                mask: false,
            });
        }
        if bounds.is_none() {
            continue;
        }
        if auto.allow_mask && diff.mask_eligible() {
            result.push(Candidate {
                strategy: if first {
                    Strategy::MaskFirst
                } else {
                    Strategy::MaskPrevious
                },
                diff: Some(Arc::clone(diff)),
                source,
                rects: vec![rect],
                mask: true,
            });
        }
        if auto.allow_multi_rect
            && let Some(rects) = regions::tiles_from_map(diff, auto.tile_size, auto.max_rectangles)?
            && rects.len() > 1
        {
            result.push(Candidate {
                strategy: if first {
                    Strategy::TilesFirst
                } else {
                    Strategy::TilesPrevious
                },
                diff: None,
                source,
                rects,
                mask: false,
            });
        }
        if auto.allow_multi_rect
            && auto.adaptive_rectangles
            && let Some(rects) =
                partition::propose(diff, auto.tile_size, auto.max_rectangles, bytes_per_pixel)?
            && rects.len() > 1
            && !result.iter().any(|candidate| {
                !candidate.mask && candidate.source == source && candidate.rects == rects
            })
        {
            merged.push(Candidate {
                strategy: if first {
                    Strategy::MergedFirst
                } else {
                    Strategy::MergedPrevious
                },
                diff: None,
                source,
                rects,
                mask: false,
            });
        }
    }
    // Keep every legacy candidate ahead of new proposals for equal-size ties.
    result.extend(merged);
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
    let mut anchor_bytes_per_pixel = 1.0;
    for (index, image) in images.iter().enumerate() {
        let analysis_started = Instant::now();
        let mut differences = Vec::new();
        let mut candidates: Vec<_> = candidates(
            images,
            index,
            auto,
            anchor_bytes_per_pixel,
            &mut differences,
        )?
        .into_iter()
        .enumerate()
        .collect();
        if auto.prune {
            candidates.sort_by_key(|(rank, candidate)| (candidate.estimated_work(), *rank));
        }
        let analysis_elapsed = analysis_started.elapsed();
        let mut completed = Vec::new();
        let mut pruned = Vec::new();
        let mut winner: Option<(usize, Vec<u8>)> = None;
        let mut winner_references = available;
        for (rank, candidate) in candidates {
            trials = trials
                .checked_add(1)
                .filter(|&n| n <= auto.max_total_trials)
                .ok_or(Error::Limit("automatic trial count"))?;
            let mut trial_refs = available;
            let started = Instant::now();
            let initial_pixels = pixels;
            let mut bytes = Vec::new();
            let mut encoded_frames = 0;
            let mut lower_bound = None;
            // Earlier canonical candidates may win equal-byte ties even if
            // cheap-first execution measured a later candidate first.
            let cutoff = if auto.prune {
                winner.as_ref().and_then(|(best_rank, best)| {
                    if rank < *best_rank {
                        best.len().checked_add(1)
                    } else {
                        Some(best.len())
                    }
                })
            } else {
                None
            };
            for (part, &rect) in candidate.rects.iter().enumerate() {
                if cutoff.is_some_and(|maximum| bytes.len() >= maximum) {
                    lower_bound = Some(bytes.len());
                    break;
                }
                pixels = pixels
                    .checked_add(u64::from(rect.width) * u64::from(rect.height))
                    .filter(|&n| n <= auto.max_trial_pixels)
                    .ok_or(Error::Limit("automatic trial pixel work"))?;
                let final_part = part + 1 == candidate.rects.len();
                let last = index + 1 == images.len() && final_part;
                let frame_image = if candidate.mask {
                    Cow::Owned(regions::mask_from_map(
                        image,
                        rect,
                        candidate.diff.as_ref().expect("mask difference map"),
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
                let budget = SectionBudget::new(
                    cutoff.map(|maximum| maximum - bytes.len()),
                    options.max_output_bytes,
                );
                match crate::encode::encode_logical_frame_trial(
                    canvas,
                    &frame,
                    options,
                    last,
                    budget.as_ref(),
                )? {
                    TrialEncode::Complete(encoded) => {
                        append(&mut bytes, encoded, options.max_output_bytes)?;
                        encoded_frames += 1;
                    }
                    TrialEncode::Bounded { lower_bound: bound } => {
                        lower_bound = Some(
                            bytes
                                .len()
                                .checked_add(bound)
                                .ok_or(Error::Limit("automatic candidate size lower bound"))?,
                        );
                        break;
                    }
                }
            }
            if let Some(lower_bound) = lower_bound {
                pruned.push((
                    rank,
                    AutoPrunedReport {
                        strategy: candidate.strategy,
                        lower_bound,
                        encoded_frames,
                        planned_frames: candidate.rects.len(),
                        trial_pixels: pixels - initial_pixels,
                        elapsed: started.elapsed(),
                    },
                ));
                continue;
            }
            completed.push((
                rank,
                AutoCandidateReport {
                    strategy: candidate.strategy,
                    encoded_bytes: bytes.len(),
                    encoded_frames,
                    trial_pixels: pixels - initial_pixels,
                    elapsed: started.elapsed(),
                },
            ));
            if winner
                .as_ref()
                .is_none_or(|(best_rank, best)| (bytes.len(), rank) < (best.len(), *best_rank))
            {
                winner_references = trial_refs;
                winner = Some((rank, bytes));
            }
        }
        let (winner_rank, winner_bytes) = winner.expect("at least one complete candidate");
        if index == 0 {
            anchor_bytes_per_pixel = winner_bytes.len() as f64
                / (u64::from(canvas.width()) * u64::from(canvas.height())) as f64;
        }
        completed.sort_by_key(|(rank, _)| *rank);
        pruned.sort_by_key(|(rank, _)| *rank);
        let winner_index = completed
            .iter()
            .position(|(rank, _)| *rank == winner_rank)
            .expect("winner has a complete report");
        append(&mut stream, winner_bytes, options.max_output_bytes)?;
        available = winner_references;
        reports.push(AutoFrameReport {
            winner_index,
            candidates: completed.into_iter().map(|(_, report)| report).collect(),
            pruned: pruned.into_iter().map(|(_, report)| report).collect(),
            differences,
            analysis_elapsed,
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout, metadata::Animation};

    #[test]
    fn reusable_auto_pool_does_not_retain_frame_references_or_trial_budgets() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let options = EncodeOptions {
            threads: 2,
            ..Default::default()
        };
        for supplied in [None, Some(&pool)] {
            for value in [3, 91, 3] {
                let images: Vec<_> = [value, value + 1]
                    .into_iter()
                    .map(|value| {
                        let image =
                            Image::new(4, 4, PixelLayout::Gray, PixelData::U8(vec![value; 16]))
                                .unwrap();
                        let mut metadata = image.metadata().clone();
                        metadata.animation = Some(Animation::default());
                        image.with_metadata(metadata, vec![]).unwrap()
                    })
                    .collect();
                let limited = AutoOptions {
                    max_total_trials: 1,
                    ..Default::default()
                };
                assert!(matches!(
                    encode_auto_in_pool(&images, &[1, 2], &options, &limited, supplied),
                    Err(Error::Limit(_))
                ));
                let result = encode_auto_in_pool(
                    &images,
                    &[1, 2],
                    &options,
                    &AutoOptions::default(),
                    supplied,
                )
                .unwrap();
                let decoded = crate::decode::decode(&result.bytes, &Default::default()).unwrap();
                assert_eq!(decoded.frames.len(), images.len());
                for (frame, image) in decoded.frames.iter().zip(&images) {
                    assert_eq!(&frame.pixels, image.pixels());
                }
                assert_eq!(result.frames.len(), images.len());
                assert!(!crate::execution::is_serial());
            }
        }
    }
}
