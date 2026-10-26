//! Driver-side commit of an already-drained, owned full/front handoff.
use super::*;
use crate::scheduler::batch::pilot_handoff::{PilotRank, PilotRequest};

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    pub(super) fn overlap_front(
        &mut self,
        automatic: &CandidateParams,
        plan: &[Candidate],
    ) -> Result<Option<PilotRank>> {
        if !self.options.scout_overlap_automatic
            || self.pilot_overlap.is_none()
            || !self.search.trials().is_empty()
            || self.probe.borrow().encodes != 0
        {
            return Ok(None);
        }
        self.search.check_batch_work(self.image, 1)?;
        let request = PilotRequest {
            image: self.image,
            automatic,
            plan,
            options: self.options,
            scan: self.scan,
            level: self.level,
            budget: TrialBudget {
                beat_bytes: None,
                max_output_bytes: self.options.full_limits.max_output_bytes,
            },
            probe_limits: self.probe.borrow().limits,
        };
        let Some(handoff) = self.pilot_overlap.as_mut().expect("checked overlap")(&request)? else {
            return Ok(None);
        };
        let usage = handoff.front.usage;
        {
            let mut work = self.probe.borrow_mut();
            work.encodes = work.encodes.saturating_add(usage.encodes);
            work.pixels = work.pixels.saturating_add(usage.pixels);
        }
        self.auxiliary_screen_batches += usage.batches;
        // Even a failed front must charge the full attempt. Do not let either
        // branch's error return before both branches have been drained.
        let mut output = Some(handoff.automatic);
        self.search.refine(
            self.image,
            9,
            automatic.clone(),
            true,
            false,
            &mut |_, _, _| output.take().expect("one automatic attempt"),
        )?;
        if usage.encodes > request.probe_limits.max_encodes
            || usage.pixels > request.probe_limits.max_encoded_pixels
        {
            return Err(Error::Limit("cumulative scheduler probe work"));
        }
        self.events.extend(handoff.front.events);
        self.events.push(SearchEvent::Route {
            name: "adaptive-scout-overlap-automatic",
            candidates: 1,
        });
        let ranked = handoff.front.ranked?;
        // Never invent a score for the speculative full, or mix coarse/fine
        // values. It remains a selectable real candidate outside this ranking.
        if ranked.is_empty()
            || ranked
                .iter()
                .any(|(_, scores)| scores.is_empty() || scores.contains(&0))
            || ranked
                .iter()
                .any(|(_, scores)| scores.len() != ranked[0].1.len())
        {
            return Err(Error::Encoding("invalid overlap pilot ranking".into()));
        }
        Ok(Some(ranked))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{
        batch::pilot_handoff::{PilotFront, PilotHandoff, PilotUsage},
        search::EncodedOutput,
    };
    use crate::{PixelData, PixelLayout};

    #[test]
    fn failed_handoffs_charge_both_branches_before_returning_error() {
        let image = Image::new(
            512,
            256,
            PixelLayout::Gray,
            PixelData::U8(vec![0; 512 * 256]),
        )
        .unwrap();
        let input = CandidateParams::default().backend(9).unwrap();
        let mut options = DriverOptions::scout();
        options.scout_pilot_first = true;
        options.scout_overlap_automatic = true;
        options.full_limits.max_full_encodes = 1;
        let no_probe =
            |_: &CandidateParams, _: &Image| -> Result<usize> { panic!("unexpected probe") };
        let no_full = |_: &CandidateParams, _: &Image, _: TrialBudget| -> Result<TrialOutput<()>> {
            panic!("unexpected full")
        };
        for failure in 0..4 {
            let screen = || {
                let callback: ProbeCallback<'_> = Box::new(no_probe);
                CandidateScreen::new(&image, None, options.screen.clone(), callback).unwrap()
            };
            let mut driver = Driver {
                image: &image,
                input: &input,
                options: &options,
                scan: None,
                level: 9,
                search: Search::new(options.full_limits),
                encode: no_full,
                full_batch: None,
                pilot_overlap: Some(Box::new(move |_| {
                    Ok(Some(PilotHandoff {
                        automatic: match failure {
                            0 => Err(Error::Encoding("injected automatic failure".into())),
                            1 => Ok(TrialOutput::Bounded { lower_bound: 100 }),
                            _ => Ok(TrialOutput::Complete(EncodedOutput {
                                bytes: vec![0; 100],
                                artifact: (),
                            })),
                        },
                        front: PilotFront {
                            ranked: if failure == 3 {
                                Ok(Vec::new())
                            } else {
                                Err(Error::Encoding("injected front failure".into()))
                            },
                            events: Vec::new(),
                            usage: PilotUsage {
                                encodes: 3,
                                pixels: 123,
                                batches: 1,
                            },
                        },
                    }))
                })),
                probe: Rc::new(RefCell::new(ProbeWork {
                    callback: no_probe,
                    limits: options.screen.limits,
                    encodes: 0,
                    pixels: 0,
                })),
                probe_batch: None,
                prefix_batch: None,
                screen: screen(),
                context_screen: screen(),
                auxiliary_screen_batches: 0,
                routes: ActiveRoutes::default(),
                calibration_feedback: Vec::new(),
                early_racing: false,
                early_family_winner: false,
                pending_automatic: false,
                analysis: ProbeResult::default(),
                frontier: Vec::new(),
                deferred: Vec::new(),
                confirmation_cache: None,
                events: Vec::new(),
            };
            assert!(driver.overlap_front(&input, &[]).is_err());
            assert!(matches!(
                driver.search.check_batch_work(&image, 1),
                Err(Error::Limit(_))
            ));
            assert_eq!(driver.probe.borrow().encodes, 3);
            assert_eq!(driver.probe.borrow().pixels, 123);
            assert_eq!(driver.auxiliary_screen_batches, 1);
            assert_eq!(driver.search.trials().len(), usize::from(failure >= 2));
        }
    }
}
