//! One candidate's execution, isolated from committed sequence reference state.
use super::append;
use crate::{
    EncodeOptions, Error, Image, Result,
    encode::budget::{SectionBudget, TrialEncode},
    frame::{BlendInfo, BlendMode, Frame, FrameOptions},
    multiframe::{
        AutoCandidateReport, AutoPrunedReport, candidates::Candidate, crop, payload::Payloads,
        references::ReferenceSlots, regions,
    },
};
use std::{borrow::Cow, time::Instant};

pub(super) enum Outcome {
    Complete {
        bytes: Vec<u8>,
        report: AutoCandidateReport,
        references: [Option<bool>; 4],
        slots: ReferenceSlots,
    },
    Pruned(AutoPrunedReport),
}

pub(super) struct Trial<'a> {
    pub reference: Option<&'a Image>,
    pub canvas: &'a Image,
    pub image: &'a Image,
    pub options: &'a EncodeOptions,
    pub index: usize,
    pub duration: u32,
    pub last_input: bool,
    pub destination: u8,
    pub max_trial_pixels: u64,
}

impl Trial<'_> {
    pub(super) fn run(
        &self,
        candidate: &Candidate,
        mut trial_refs: [Option<bool>; 4],
        mut trial_slots: ReferenceSlots,
        pixels: &mut u64,
        payloads: &mut Payloads,
        cutoff: Option<usize>,
    ) -> Result<Outcome> {
        let Self {
            canvas,
            image,
            options,
            index,
            destination,
            ..
        } = *self;
        let final_save = if self.last_input { 0 } else { destination };
        let started = Instant::now();
        let initial_pixels = *pixels;
        let workers = crate::execution::worker_count().min(4);
        if (cutoff.is_none()
            || (self.max_trial_pixels == u64::MAX && options.max_output_bytes == usize::MAX))
            && candidate.replace()
            && candidate.rects.len() > 1
            && workers > 1
            && !crate::execution::is_serial()
            && rayon::current_thread_index().is_some()
        {
            return self.run_parallel(
                candidate,
                trial_refs,
                trial_slots,
                pixels,
                payloads,
                if cutoff.is_some() {
                    workers.min(2)
                } else {
                    workers
                },
                started,
                cutoff,
            );
        }
        let mut bytes = Vec::new();
        let mut encoded_frames = 0;
        let mut submitted_parts = 0;
        let mut lower_bound = None;
        let planned_frames = if candidate.multiply_mask() {
            2
        } else {
            candidate.rects.len()
        };
        let mut multiply_images: Option<std::array::IntoIter<Image, 2>> = None;
        // A partial Replace copies fresh samples into the rectangle and retains
        // its background everywhere else. Capture the bound before any parts
        // overwrite the source slot (source and destination may coincide).
        let source_depth = trial_slots.add_depth[usize::from(candidate.source)];
        let mut depth = source_depth;
        for part in 0..planned_frames {
            let rect = candidate.rects[if candidate.multiply_mask() { 0 } else { part }];
            if cutoff.is_some_and(|maximum| bytes.len() >= maximum) {
                lower_bound = Some(bytes.len());
                break;
            }
            *pixels = pixels
                .checked_add(u64::from(rect.width) * u64::from(rect.height))
                .filter(|&n| n <= self.max_trial_pixels)
                .ok_or(Error::Limit("automatic trial pixel work"))?;
            submitted_parts += 1;
            let final_part = part + 1 == planned_frames;
            let last = self.last_input && final_part;
            let delta = candidate.strategy == crate::multiframe::Strategy::AddDelta;
            let frame_image = if delta {
                Cow::Owned(crate::multiframe::add_delta::image(
                    self.reference.expect("delta reference"),
                    image,
                    rect,
                    candidate.diff.as_ref().expect("delta change map"),
                )?)
            } else if candidate.multiply_mask() {
                if multiply_images.is_none() {
                    let (keep, update) = crate::multiframe::multiply_mask::images(
                        image,
                        rect,
                        candidate
                            .diff
                            .as_ref()
                            .expect("multiply mask difference map"),
                    )?;
                    multiply_images = Some([keep, update].into_iter());
                }
                Cow::Owned(
                    multiply_images
                        .as_mut()
                        .expect("multiply mask images")
                        .next()
                        .expect("two multiply mask layers"),
                )
            } else if candidate.mask {
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
                    duration: if final_part { self.duration } else { 0 },
                    save_as_reference: if last { 0 } else { destination },
                    blend: BlendInfo {
                        source: if part == 0 {
                            candidate.source
                        } else {
                            destination
                        },
                        mode: if delta {
                            BlendMode::Add
                        } else if candidate.multiply_mask() {
                            if part == 0 {
                                BlendMode::Multiply
                            } else {
                                BlendMode::Add
                            }
                        } else if candidate.mask {
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
            let mut generated_atlas = false;
            let full = candidate.strategy == crate::multiframe::Strategy::Full;
            let encoded = if full {
                crate::encode::encode_replace_frame_trial(
                    canvas,
                    &frame,
                    options,
                    last,
                    budget.as_ref(),
                    &mut generated_atlas,
                )
            } else if !candidate.replace() {
                crate::encode::encode_logical_frame_trial(
                    canvas,
                    &frame,
                    options,
                    last,
                    budget.as_ref(),
                )
            } else {
                let (encoded, atlas) =
                    payloads.encode(rect, canvas, &frame, options, last, budget.as_ref())?;
                generated_atlas = atlas;
                Ok(encoded)
            }?;
            match encoded {
                TrialEncode::Complete(encoded) => {
                    append(&mut bytes, encoded, options.max_output_bytes)?;
                    encoded_frames += 1;
                    depth = if generated_atlas {
                        None
                    } else if full {
                        Some(0)
                    } else if delta {
                        source_depth.map(|n| n + 1)
                    } else if candidate.replace() || candidate.multiply_mask() {
                        // Multiply by exact 0/1, followed by adding fresh samples
                        // to cleared locations and zero to all other samples.
                        depth
                    } else {
                        None
                    };
                    trial_slots.saved_layer(
                        index,
                        frame.options.save_as_reference,
                        frame.options.duration,
                        final_part,
                        last,
                    );
                    if final_part && !last {
                        trial_slots.add_depth[usize::from(destination)] = depth;
                    }
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
            if candidate.replace() {
                payloads.skip(&candidate.rects[submitted_parts..]);
            }
            return Ok(Outcome::Pruned(AutoPrunedReport {
                strategy: candidate.strategy,
                tile_size: candidate.tile_size,
                reference_index: candidate.reference_index,
                source: candidate.source,
                save_as_reference: final_save,
                lower_bound,
                encoded_frames,
                planned_frames,
                trial_pixels: *pixels - initial_pixels,
                elapsed: started.elapsed(),
            }));
        }
        let report = AutoCandidateReport {
            strategy: candidate.strategy,
            tile_size: candidate.tile_size,
            reference_index: candidate.reference_index,
            source: candidate.source,
            save_as_reference: final_save,
            encoded_bytes: bytes.len(),
            encoded_frames,
            trial_pixels: *pixels - initial_pixels,
            elapsed: started.elapsed(),
        };
        Ok(Outcome::Complete {
            bytes,
            report,
            references: trial_refs,
            slots: trial_slots,
        })
    }

    /// Replace payloads enter this path. Bounded trials require unlimited hard
    /// budgets and use at most two speculative parts. Images and
    /// encoded payloads live for at most one worker-sized batch, never the
    /// entire candidate. Reference validation/publication stays canonical.
    #[allow(clippy::too_many_arguments)]
    fn run_parallel(
        &self,
        candidate: &Candidate,
        mut references: [Option<bool>; 4],
        mut slots: ReferenceSlots,
        pixels: &mut u64,
        payloads: &mut Payloads,
        workers: usize,
        started: Instant,
        cutoff: Option<usize>,
    ) -> Result<Outcome> {
        let initial_pixels = *pixels;
        let mut bytes = Vec::new();
        let mut encoded_frames = 0;
        let mut lower_bound = None;
        let mut submitted_parts = 0;
        let mut depth = slots.add_depth[usize::from(candidate.source)];
        for (batch_index, rects) in candidate.rects.chunks(workers).enumerate() {
            if cutoff.is_some_and(|maximum| bytes.len() >= maximum) {
                lower_bound = Some(bytes.len());
                break;
            }
            let remaining = cutoff.map(|maximum| maximum - bytes.len());
            let first = batch_index * workers;
            let mut jobs = Vec::with_capacity(rects.len());
            // Charge all speculative pixels before dispatch, so workers can
            // never collectively exceed the existing cumulative work cap.
            for (i, &rect) in rects.iter().enumerate() {
                *pixels = pixels
                    .checked_add(u64::from(rect.width) * u64::from(rect.height))
                    .filter(|&n| n <= self.max_trial_pixels)
                    .ok_or(Error::Limit("automatic trial pixel work"))?;
                let part = first + i;
                let final_part = part + 1 == candidate.rects.len();
                let last = self.last_input && final_part;
                let options = FrameOptions {
                    origin: (rect.x as i32, rect.y as i32),
                    duration: if final_part { self.duration } else { 0 },
                    save_as_reference: if last { 0 } else { self.destination },
                    blend: BlendInfo {
                        source: if part == 0 {
                            candidate.source
                        } else {
                            self.destination
                        },
                        mode: BlendMode::Replace,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                jobs.push((options, last, final_part, rect, payloads.take_rect(rect)));
                submitted_parts += 1;
            }
            let encoded =
                crate::execution::map_mut(&mut jobs, |_, (options, last, _, rect, cache)| {
                    let result = (|| {
                        let image = crop(self.image, *rect)?;
                        let budget = SectionBudget::new(remaining, self.options.max_output_bytes);
                        cache.encode(
                            *rect,
                            self.canvas,
                            &Frame {
                                image: &image,
                                options: options.clone(),
                            },
                            self.options,
                            *last,
                            budget.as_ref(),
                        )
                    })();
                    if result.is_err() {
                        // Failed encoding has not consumed its cache use.
                        cache.skip(std::slice::from_ref(rect));
                    }
                    result
                });
            // Restore every cache before inspecting results. This includes
            // speculative results beyond a cutoff and all error paths.
            for (_, _, _, _, cache) in jobs.iter_mut() {
                payloads.restore_rect(std::mem::take(cache));
            }
            for (result, (settings, last, final_part, rect, _)) in encoded.into_iter().zip(jobs) {
                if cutoff.is_some_and(|maximum| bytes.len() >= maximum) {
                    lower_bound = Some(bytes.len());
                    break;
                }
                let (output, atlas) = result?;
                references = crate::encode::validate_frame_references(
                    self.canvas.into(),
                    // crop() retains an exact clone of the input metadata.
                    crate::frame::Geometry {
                        width: rect.width,
                        height: rect.height,
                        metadata: self.image.metadata(),
                    },
                    &settings,
                    self.options,
                    last,
                    references,
                )?;
                if atlas {
                    depth = None;
                }
                let output = match output {
                    TrialEncode::Complete(output) => output,
                    TrialEncode::Bounded { lower_bound: bound } => {
                        lower_bound = Some(
                            bytes
                                .len()
                                .checked_add(bound)
                                .ok_or(Error::Limit("automatic candidate size lower bound"))?,
                        );
                        break;
                    }
                };
                append(&mut bytes, output, self.options.max_output_bytes)?;
                encoded_frames += 1;
                slots.saved_layer(
                    self.index,
                    settings.save_as_reference,
                    settings.duration,
                    final_part,
                    last,
                );
                if final_part && !last {
                    slots.add_depth[usize::from(self.destination)] = depth;
                }
            }
            if lower_bound.is_some() {
                break;
            }
        }
        if let Some(lower_bound) = lower_bound {
            payloads.skip(&candidate.rects[submitted_parts..]);
            return Ok(Outcome::Pruned(AutoPrunedReport {
                strategy: candidate.strategy,
                tile_size: candidate.tile_size,
                reference_index: candidate.reference_index,
                source: candidate.source,
                save_as_reference: if self.last_input { 0 } else { self.destination },
                lower_bound,
                encoded_frames,
                planned_frames: candidate.rects.len(),
                trial_pixels: *pixels - initial_pixels,
                elapsed: started.elapsed(),
            }));
        }
        let report = AutoCandidateReport {
            strategy: candidate.strategy,
            tile_size: candidate.tile_size,
            reference_index: candidate.reference_index,
            source: candidate.source,
            save_as_reference: if self.last_input { 0 } else { self.destination },
            encoded_bytes: bytes.len(),
            encoded_frames,
            trial_pixels: *pixels - initial_pixels,
            elapsed: started.elapsed(),
        };
        Ok(Outcome::Complete {
            bytes,
            report,
            references,
            slots,
        })
    }
}
