//! Per-logical-frame search using real encoded layers, with retained winners.
mod trial;

use super::{
    AutoEncodeResult, AutoFrameReport, AutoOptions, Reference, candidates::candidates, compatible,
    lookahead::Lookahead, payload::Payloads, references::ReferenceSlots,
};
use crate::{EncodeOptions, Error, Image, Result};
use std::time::Instant;
use trial::{Outcome, Trial};

struct Winner {
    rank: usize,
    bytes: Vec<u8>,
    references: [Option<bool>; 4],
    slots: ReferenceSlots,
}

impl Winner {
    fn cutoff(&self, rank: usize) -> Option<usize> {
        // Earlier canonical candidates may still win an equal-byte tie.
        if rank < self.rank {
            self.bytes.len().checked_add(1)
        } else {
            Some(self.bytes.len())
        }
    }
}

/// Encode exact full-image states. Candidates share metadata, displayed pixels
/// and timing. Commit the winner's bytes AND reference state without re-encoding;
/// retained-reference candidates may choose different destination slots.
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
    if auto.extra_tile_sizes.contains(&0) {
        return Err(Error::InvalidInput(
            "extra_tile_sizes must be positive".into(),
        ));
    }
    if auto.retained_reference && auto.reference != Reference::SmallestCrop {
        return Err(Error::InvalidInput(
            "retained_reference requires the automatic first/previous reference mode".into(),
        ));
    }
    if auto.reference_lookahead && !auto.retained_reference {
        return Err(Error::InvalidInput(
            "reference_lookahead requires retained_reference".into(),
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
    let mut slots = ReferenceSlots::default();
    let mut lookahead = Lookahead::default();
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
            &slots,
            &lookahead,
        )?
        .into_iter()
        .enumerate()
        .collect();
        lookahead = if auto.reference_lookahead {
            Lookahead::prepare(images, index, &slots)?
        } else {
            Lookahead::default()
        };
        if auto.prune {
            candidates.sort_by_key(|(rank, candidate)| (candidate.estimated_work(), *rank));
        }
        let mut payloads = Payloads::new(candidates.iter().map(|(_, candidate)| candidate));
        let analysis_elapsed = analysis_started.elapsed();
        let mut completed = Vec::new();
        let mut pruned = Vec::new();
        let mut winner: Option<Winner> = None;
        for (rank, candidate) in candidates {
            trials = trials
                .checked_add(1)
                .filter(|&n| n <= auto.max_total_trials)
                .ok_or(Error::Limit("automatic trial count"))?;
            let destination = lookahead
                .keep
                .map(|keep| if keep == 1 { 2 } else { 1 })
                .unwrap_or_else(|| {
                    slots.destination(
                        index,
                        candidate.reference_index.map(|_| candidate.source),
                        auto.retained_reference,
                    )
                });
            let cutoff = if auto.prune {
                winner.as_ref().and_then(|best| best.cutoff(rank))
            } else {
                None
            };
            let trial = Trial {
                canvas,
                image,
                options,
                index,
                duration: durations[index],
                last_input: index + 1 == images.len(),
                destination,
                max_trial_pixels: auto.max_trial_pixels,
            };
            match trial.run(
                &candidate,
                available,
                slots,
                &mut pixels,
                &mut payloads,
                cutoff,
            )? {
                Outcome::Pruned(report) => pruned.push((rank, report)),
                Outcome::Complete {
                    bytes,
                    report,
                    references,
                    slots,
                } => {
                    completed.push((rank, report));
                    if winner
                        .as_ref()
                        .is_none_or(|best| (bytes.len(), rank) < (best.bytes.len(), best.rank))
                    {
                        winner = Some(Winner {
                            rank,
                            bytes,
                            references,
                            slots,
                        });
                    }
                }
            }
        }
        let Winner {
            rank: winner_rank,
            bytes: winner_bytes,
            references: winner_references,
            slots: winner_slots,
        } = winner.expect("at least one complete candidate");
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
            retained_reference: auto.retained_reference,
            reference_lookahead: auto.reference_lookahead,
            lookahead_comparisons: lookahead.comparisons,
            reused_payloads: payloads.hits,
            reused_payload_pixels: payloads.pixels,
            references_before: slots.frames,
            references_after: winner_slots.frames,
            winner_index,
            candidates: completed.into_iter().map(|(_, report)| report).collect(),
            pruned: pruned.into_iter().map(|(_, report)| report).collect(),
            differences,
            analysis_elapsed,
        });
        slots = winner_slots;
        lookahead.retain(&slots);
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
mod tests;
