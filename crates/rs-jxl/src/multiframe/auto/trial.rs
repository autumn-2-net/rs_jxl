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
        let mut bytes = Vec::new();
        let mut encoded_frames = 0;
        let mut submitted_parts = 0;
        let mut lower_bound = None;
        for (part, &rect) in candidate.rects.iter().enumerate() {
            if cutoff.is_some_and(|maximum| bytes.len() >= maximum) {
                lower_bound = Some(bytes.len());
                break;
            }
            *pixels = pixels
                .checked_add(u64::from(rect.width) * u64::from(rect.height))
                .filter(|&n| n <= self.max_trial_pixels)
                .ok_or(Error::Limit("automatic trial pixel work"))?;
            submitted_parts += 1;
            let final_part = part + 1 == candidate.rects.len();
            let last = self.last_input && final_part;
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
                    duration: if final_part { self.duration } else { 0 },
                    save_as_reference: if last { 0 } else { destination },
                    blend: BlendInfo {
                        source: if part == 0 {
                            candidate.source
                        } else {
                            destination
                        },
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
            let encoded = if candidate.mask {
                crate::encode::encode_logical_frame_trial(
                    canvas,
                    &frame,
                    options,
                    last,
                    budget.as_ref(),
                )
            } else {
                payloads.encode(rect, canvas, &frame, options, last, budget.as_ref())
            }?;
            match encoded {
                TrialEncode::Complete(encoded) => {
                    append(&mut bytes, encoded, options.max_output_bytes)?;
                    encoded_frames += 1;
                    trial_slots.saved_layer(
                        index,
                        frame.options.save_as_reference,
                        frame.options.duration,
                        final_part,
                        last,
                    );
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
            if !candidate.mask {
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
}
