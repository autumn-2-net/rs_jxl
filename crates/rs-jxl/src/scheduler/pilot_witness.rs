//! Full-image confirmations for representation changes hidden by crop ranking.
use super::*;
use crate::modular::{
    predictor::Predictor,
    source_options::{SourceBuffering, one_pass_geometry},
};

pub(crate) fn candidates(
    image: &Image,
    input: &CandidateParams,
    leader: &CandidateParams,
    scan: Option<&LosslessScan>,
) -> Result<Vec<CandidateParams>> {
    let mut result = Vec::new();
    if !scan.is_some_and(|s| s.literal_candidate) {
        return Ok(result);
    }
    if image.bits() > 8 {
        // Wide integer tuples and their RCT residuals have very different
        // palette capacity; a narrow band can reverse their full-image order.
        let palette = literal(input, 9, 0.01)?;
        if palette != *leader {
            result.push(palette);
        }
        let mut raw = input.backend(9)?;
        raw.predictor = Some(Predictor::Zero);
        raw.group_size_shift = Some(3);
        raw.rct = Some(6);
        raw.palette_colors = 0;
        raw.channel_colors_percent = 0.0;
        raw.channel_colors_pre_transform_percent = 0.0;
        raw.learning_fraction = 0.1;
        raw.max_properties = 0;
        raw.patches = super::super::params::Override::Off;
        result.push(raw);
    }
    if leader.predictor == Some(Predictor::Zero)
        && matches!(
            input.buffering,
            SourceBuffering::Auto | SourceBuffering::FullFrame
        )
    {
        let mut global = winner_neighborhood(&input.backend(9)?, image.channel_count());
        // Only add the global witness when normal geometry permits streaming.
        // Explicit streaming requests remain untouched.
        if one_pass_geometry(image.width(), image.height(), global.group_size_shift)?.num_groups > 8
        {
            global.buffering = SourceBuffering::FullFrame;
            if global != *leader {
                result.push(global);
            }
        }
    }
    Ok(result)
}

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    pub(super) fn pilot_representation_witness(&mut self, leader: &CandidateParams) -> Result<()> {
        if !self.options.scout_representation_witness {
            return Ok(());
        }
        let plan = candidates(self.image, self.input, leader, self.scan)?;
        let mut completed = 0;
        for candidate in plan {
            if self.search.find(&candidate).is_none() {
                let baseline = self.search.trials().is_empty();
                self.full(9, candidate, baseline, false)?;
                completed += 1;
            }
        }
        if completed != 0 {
            self.events.push(SearchEvent::Route {
                name: "adaptive-scout-representation-witness",
                candidates: completed,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout};

    #[test]
    fn real_palette_witness_survives_misleading_probe_ranking() {
        use crate::scheduler::search::EncodedOutput;
        let image = Image::new(
            512,
            256,
            PixelLayout::Rgb,
            PixelData::U16(vec![65535; 512 * 256 * 3]),
        )
        .unwrap();
        let input = CandidateParams::default().backend(9).unwrap();
        let mut options = DriverOptions::scout();
        options.scout_pilot_first = true;
        options.scout_representation_witness = true;
        options.coverage_scan = true;
        options.preflight.enabled = true;
        options.screen.adaptive_coverage = true;
        let mut calls = Vec::new();
        let output = run_pixel_search(
            &image,
            &input,
            &options,
            |params, _, budget| {
                calls.push((params.predictor, budget.beat_bytes));
                let size = if params.predictor == Some(Predictor::Zero) && params.palette_colors > 0
                {
                    40
                } else {
                    100
                };
                if budget.beat_bytes.is_some_and(|bound| size >= bound) {
                    Ok(TrialOutput::Bounded { lower_bound: size })
                } else {
                    Ok(TrialOutput::Complete(EncodedOutput {
                        bytes: vec![17; size],
                        artifact: size,
                    }))
                }
            },
            |params, _| {
                Ok(if params.predictor == Some(Predictor::Variable) {
                    10
                } else {
                    30
                })
            },
        )
        .unwrap();
        assert_eq!(output.bytes.len(), 40);
        assert_eq!(output.artifact, 40);
        assert_eq!(output.params.predictor, Some(Predictor::Zero));
        assert_eq!(output.params.speed_tier, 1);
        assert_eq!(calls[0], (Some(Predictor::Variable), None));
        assert_eq!(calls[1], (Some(Predictor::Zero), Some(100)));
        assert!(output.report.trials.iter().any(|t| t.lower_bound.is_some()));
        assert!(output.report.trials[output.report.winner_index].bytes == Some(40));
    }

    #[test]
    fn wide_palette_and_raw_witnesses_keep_effort_and_exact_integer_input() {
        let image = Image::new(8, 8, PixelLayout::Rgb, PixelData::U16(vec![65535; 192])).unwrap();
        let input = CandidateParams::default().backend(9).unwrap();
        let leader = CandidateParams {
            predictor: Some(Predictor::Variable),
            ..input.clone()
        };
        let scan = LosslessScan {
            literal_candidate: true,
            ..Default::default()
        };
        let plan = candidates(&image, &input, &leader, Some(&scan)).unwrap();
        assert_eq!(plan.len(), 2);
        assert!(
            plan.iter()
                .all(|p| p.speed_tier == 1 && p.predictor == Some(Predictor::Zero))
        );
        assert_eq!(plan[0].palette_colors, 10000);
        assert_eq!(plan[1].palette_colors, 0);
        assert_eq!(plan[1].rct, Some(6));
        assert_eq!(plan[1].learning_fraction, 0.1);
        assert!(
            candidates(&image, &input, &leader, None)
                .unwrap()
                .is_empty()
        );
        let natural = LosslessScan {
            literal_candidate: false,
            ..scan
        };
        assert!(
            candidates(&image, &input, &leader, Some(&natural))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn global_witness_requires_palette_leader_and_respects_explicit_streaming() {
        let image = Image::new(
            1536,
            1536,
            PixelLayout::Gray,
            PixelData::U8(vec![7; 1536 * 1536]),
        )
        .unwrap();
        let input = CandidateParams::default().backend(9).unwrap();
        let leader = literal(&input, 9, 0.01).unwrap();
        let scan = LosslessScan {
            literal_candidate: true,
            ..Default::default()
        };
        let plan = candidates(&image, &input, &leader, Some(&scan)).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].buffering, SourceBuffering::FullFrame);
        assert_eq!(plan[0].predictor, None);
        assert_eq!(plan[0].group_size_shift, Some(2));
        assert_eq!(plan[0].speed_tier, 1);
        for buffering in [
            SourceBuffering::Streaming,
            SourceBuffering::StreamingAbove2048,
        ] {
            let explicit = CandidateParams {
                buffering,
                ..input.clone()
            };
            assert!(
                candidates(&image, &explicit, &leader, Some(&scan))
                    .unwrap()
                    .is_empty()
            );
        }
        assert!(
            candidates(&image, &input, &input, Some(&scan))
                .unwrap()
                .is_empty()
        );
        let small = Image::new(8, 8, PixelLayout::Gray, PixelData::U8(vec![7; 64])).unwrap();
        assert!(
            candidates(&small, &input, &leader, Some(&scan))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn competitive_palette_contender_keeps_global_witness_despite_cross_channel_crop_leader() {
        use crate::scheduler::search::EncodedOutput;
        for effort in [9, 10] {
            let image = Image::new(
                1536,
                1536,
                PixelLayout::Gray,
                PixelData::U8(vec![7; 1536 * 1536]),
            )
            .unwrap();
            let mut options = DriverOptions::scout();
            options.scout_pilot_first = true;
            options.scout_representation_witness = true;
            options.coverage_scan = true;
            let mut global_calls = 0;
            let output = run_pixel_search(
                &image,
                &CandidateParams::default().backend(effort).unwrap(),
                &options,
                |params, _, budget| {
                    let global = params.buffering == SourceBuffering::FullFrame
                        && params.predictor.is_none()
                        && params.group_size_shift == Some(2);
                    global_calls += usize::from(global);
                    let size = if global { 40 } else { 100 };
                    Ok(if budget.beat_bytes.is_some_and(|bound| size >= bound) {
                        TrialOutput::Bounded { lower_bound: size }
                    } else {
                        TrialOutput::Complete(EncodedOutput {
                            bytes: vec![17; size],
                            artifact: global,
                        })
                    })
                },
                |params, _| {
                    Ok(match params.predictor {
                        Some(Predictor::Variable) => 100,
                        Some(Predictor::Zero) => 101,
                        _ => 300,
                    })
                },
            )
            .unwrap();
            assert!(output.artifact);
            assert_eq!(output.bytes.len(), 40);
            assert_eq!(output.params.buffering, SourceBuffering::FullFrame);
            assert_eq!(global_calls, if effort == 9 { 1 } else { 2 });
        }
    }
}
