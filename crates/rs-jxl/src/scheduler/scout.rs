//! Experimental coverage-guided frontier. Only actual full-frame outputs can win.
use super::*;

/// A rival gets a full-frame witness if it is close globally or clearly better
/// in a material region. Small flat crops must not dominate this decision.
fn regional_rival(candidate: &[usize], leader: &[usize]) -> bool {
    if candidate.is_empty() || candidate.len() != leader.len() {
        return true;
    }
    let a: f64 = candidate.iter().map(|&n| n as f64).sum();
    let b: f64 = leader.iter().map(|&n| n as f64).sum();
    b == 0.0
        || a <= b * 1.025
        || candidate
            .iter()
            .zip(leader)
            .any(|(&a, &b_i)| b_i as f64 >= b * 0.10 && (a as f64) < b_i as f64 * 0.95)
}

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    pub(super) fn run_scout(&mut self) -> Result<bool> {
        if !self.options.adaptive_scout || !refinable(self.image) {
            return Ok(false);
        }
        if self.run_compact_pilot()? {
            return Ok(true);
        }
        if !self.screen.available() {
            return Ok(false);
        }
        let _scope = crate::profile::scope("scheduler.scout");
        if self.options.scout_pilot_first {
            return self.run_scout_pilot();
        }
        let initial =
            PixelSearch::new(self.image, self.input.clone(), self.scan)?.initial_plan()?;
        let sizes = self.initial_full(&initial)?;
        let regular = initial
            .iter()
            .zip(&sizes)
            .find(|(request, _)| request.stage == 1)
            .and_then(|(_, size)| *size);
        let palette = initial
            .iter()
            .zip(&sizes)
            .find(|(request, _)| request.family == super::super::policy::TrialFamily::Literal)
            .and_then(|(_, size)| *size);
        let palette_guard =
            matches!((regular, palette), (Some(r), Some(p)) if p as f64 <= r as f64 * 0.85);
        // Crop headers/tree overhead can hide full-image MA gains in a palette
        // stream. A measured learned witness closes that capacity mismatch.
        let mut witnesses = 0;
        if palette_guard {
            let learned_params = literal(self.input, 9, 0.01)?;
            let learned = self.full(3, learned_params.clone(), false, false)?;
            witnesses += 1;
            // This is a search-budget heuristic, not proof that unseen variants
            // cannot win. Keep deeper exploration available at the next level.
            if matches!((palette, learned), (Some(p), Some(l)) if l as f64 <= p as f64 * 0.95) {
                // Strong within-family improvement is not cross-family evidence.
                // Compare the two cross-channel representations at full probe
                // fidelity before admitting one real full-frame challenger.
                let control = self.screen.score(&learned_params)?;
                let mut rivals = Vec::new();
                for candidate in lossless_probe_plan(self.input, self.options.plan)
                    .into_iter()
                    .filter(|c| {
                        c.params.predictor == Some(crate::modular::predictor::Predictor::Variable)
                    })
                {
                    let scores = self.screen.score(&candidate.params)?;
                    let sum = total(&scores)?;
                    if crate::profile::enabled() {
                        crate::profile::note(
                            "scheduler.scout_rival",
                            format_args!(
                                "rct={:?} control={control:?} scores={scores:?} competitive={}",
                                candidate.params.rct,
                                regional_rival(&scores, &control)
                            ),
                        );
                    }
                    rivals.push((candidate.params, sum, regional_rival(&scores, &control)));
                }
                rivals.sort_by_key(|(_, score, _)| *score);
                if let Some((params, _, _)) = rivals.iter().find(|(_, _, competitive)| *competitive)
                {
                    let before = self.best_size(4)?;
                    let size = self.full(3, params.clone(), false, false)?;
                    witnesses += 1;
                    // A real family switch justifies one sibling confirmation:
                    // RCT ordering on small crops need not match the full image.
                    if size.is_some_and(|n| n < before)
                        && let Some((sibling, _, _)) = rivals.iter().find(|(p, _, _)| p != params)
                    {
                        self.full(3, sibling.clone(), false, false)?;
                        witnesses += 1;
                    }
                }
                if self.level == 9 {
                    self.events.push(SearchEvent::Route {
                        name: "adaptive-scout-calibrated-palette",
                        candidates: witnesses,
                    });
                    return Ok(true);
                }
            }
        }
        let anchor = self.best_params(4)?;
        let mut plans = lossless_probe_plan(self.input, self.options.plan);
        for effort in [7, 8, 9] {
            plans.push(Candidate::new("automatic", self.input.backend(effort)?));
        }
        plans.push(Candidate::new(
            "anchor-neighborhood",
            winner_neighborhood(&anchor, self.image.channel_count()),
        ));
        for fraction in [0.0, 0.01, 0.03, 0.1, 1.0] {
            plans.push(Candidate::new("literal", literal(self.input, 9, fraction)?));
        }
        let ranked = self.screen.rank(plans, &anchor, true)?;
        let leader = ranked
            .first()
            .ok_or_else(|| Error::Encoding("empty scout frontier".into()))?;
        let mut leader_scores = self.screen.score(if palette_guard {
            &anchor
        } else {
            &leader.params
        })?;
        self.events.push(SearchEvent::Frontier {
            candidates: ranked.clone(),
        });
        let mut selected = witnesses;
        let cap = if self.level == 10 { 5 } else { 3 };
        for candidate in ranked {
            if self.search.find(&candidate.params).is_some() {
                continue;
            }
            let scores = self.screen.score(&candidate.params)?;
            if selected != 0 && !regional_rival(&scores, &leader_scores) {
                continue;
            }
            let before = self.best_size(10)?;
            // This is a complete encode (or a proven byte bound), never a probe
            // estimate substituted for a real output or a timed-out attempt.
            let size = self.full(9, candidate.params.clone(), false, false)?;
            selected += 1;
            if size.is_some_and(|n| n < before) {
                leader_scores = scores;
            }
            if selected == cap {
                break;
            }
        }
        if self.level == 10 {
            let winner = self.best_params(10)?;
            for params in [winner.backend(10)?, self.input.backend(10)?] {
                if self.search.find(&params).is_none() {
                    self.full(10, params, false, false)?;
                }
            }
        }
        self.events.push(SearchEvent::Route {
            name: "adaptive-scout",
            candidates: selected,
        });
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_width_probe_override_is_scoped_to_scout() {
        use crate::scheduler::search::EncodedOutput;
        use crate::{PixelData, PixelLayout};
        let image = Image::new(
            1024,
            512,
            PixelLayout::Gray,
            PixelData::U8(vec![9; 1024 * 512]),
        )
        .unwrap();
        let input = CandidateParams::default().backend(9).unwrap();
        for scout in [false, true] {
            let mut options = if scout {
                DriverOptions::scout()
            } else {
                DriverOptions::default()
            };
            options.screen.regions.full_width_strips = true;
            let output = run_pixel_search(
                &image,
                &input,
                &options,
                |_, _, _| {
                    Ok(TrialOutput::Complete(EncodedOutput {
                        bytes: vec![0; 100],
                        artifact: (),
                    }))
                },
                |_, sample| {
                    assert_eq!(sample.width(), 1024);
                    Ok(100)
                },
            );
            assert_eq!(output.is_ok(), scout);
        }
    }

    #[test]
    fn regional_agreement_keeps_close_or_material_local_winners() {
        assert!(regional_rival(&[102, 102, 102], &[100, 100, 100]));
        assert!(!regional_rival(&[110, 110, 110], &[100, 100, 100]));
        assert!(regional_rival(&[90, 130, 130], &[100, 100, 100]));
        assert!(!regional_rival(&[1, 150, 150], &[2, 100, 100]));
        assert!(regional_rival(&[], &[]));
        assert!(regional_rival(&[0], &[0]));
        assert!(regional_rival(&[10], &[10, 10]));
    }

    #[test]
    fn scout_retains_only_real_outputs_and_higher_budget_keeps_prefix() {
        use crate::scheduler::search::EncodedOutput;
        use crate::{PixelData, PixelLayout};
        let image = Image::new(
            1024,
            512,
            PixelLayout::Gray,
            PixelData::U8(vec![9; 1024 * 512]),
        )
        .unwrap();
        let mut sizes = Vec::new();
        for effort in [9, 10] {
            let mut calls = 0;
            let result = run_pixel_search(
                &image,
                &CandidateParams::default().backend(effort).unwrap(),
                &DriverOptions::scout(),
                |params, _, budget| {
                    calls += 1;
                    let size = if params.predictor
                        == Some(crate::modular::predictor::Predictor::Weighted)
                    {
                        80
                    } else {
                        100
                    };
                    if budget.beat_bytes.is_some_and(|bound| size >= bound) {
                        return Ok(TrialOutput::Bounded { lower_bound: size });
                    }
                    Ok(TrialOutput::Complete(EncodedOutput {
                        bytes: vec![calls as u8; size],
                        artifact: calls,
                    }))
                },
                |params, _| {
                    Ok(
                        if params.predictor == Some(crate::modular::predictor::Predictor::Weighted)
                        {
                            10
                        } else {
                            20
                        },
                    )
                },
            )
            .unwrap();
            assert_eq!(result.bytes, vec![result.artifact as u8; 80]);
            assert!(result.report.events.iter().any(|event| matches!(
                event,
                SearchEvent::Route {
                    name: "adaptive-scout",
                    ..
                }
            )));
            assert!(result.report.probe_encodes > 0);
            sizes.push(result.bytes.len());
        }
        assert!(sizes[1] <= sizes[0]);
    }

    #[test]
    fn invalid_scout_effort_fails_before_encoding() {
        use crate::{PixelData, PixelLayout};
        let image = Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![0])).unwrap();
        let result = run_pixel_search::<_, _, ()>(
            &image,
            &CandidateParams::default().backend(8).unwrap(),
            &DriverOptions::scout(),
            |_, _, _| panic!("invalid options reached full encoder"),
            |_, _| panic!("invalid options reached probe encoder"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn learned_palette_witness_survives_misleading_crop_scores() {
        use crate::scheduler::search::EncodedOutput;
        use crate::{PixelData, PixelLayout, modular::predictor::Predictor};
        let image = Image::new(
            1024,
            512,
            PixelLayout::Gray,
            PixelData::U8(vec![9; 1024 * 512]),
        )
        .unwrap();
        let input = CandidateParams::default().backend(9).unwrap();
        let learned = literal(&input, 9, 0.01).unwrap();
        let mut saw_witness = false;
        let output = run_pixel_search(
            &image,
            &input,
            &DriverOptions::scout(),
            |params, _, budget| {
                let size = if *params == learned {
                    saw_witness = true;
                    200
                } else if params.predictor == Some(Predictor::Zero) {
                    300
                } else {
                    1000
                };
                if budget.beat_bytes.is_some_and(|bound| size >= bound) {
                    return Ok(TrialOutput::Bounded { lower_bound: size });
                }
                Ok(TrialOutput::Complete(EncodedOutput {
                    bytes: vec![7; size],
                    artifact: (),
                }))
            },
            |params, _| {
                Ok(if params.learning_fraction == 0.0 {
                    10
                } else {
                    100
                })
            },
        )
        .unwrap();
        assert!(saw_witness);
        assert_eq!(output.bytes.len(), 200);
        assert_eq!(output.params, learned);
        assert_eq!(output.report.full_encodes, 4);
        assert!(output.report.probe_encodes > 0);
    }

    #[test]
    fn palette_gain_does_not_hide_a_cross_channel_winner_at_either_effort() {
        use crate::scheduler::search::EncodedOutput;
        use crate::{PixelData, PixelLayout, modular::predictor::Predictor};
        let image = Image::new(
            1024,
            512,
            PixelLayout::Gray,
            PixelData::U8(vec![9; 1024 * 512]),
        )
        .unwrap();
        for effort in [9, 10] {
            let input = CandidateParams::default().backend(effort).unwrap();
            let learned = literal(&input, 9, 0.01).unwrap();
            let cross =
                |p: &CandidateParams| p.predictor == Some(Predictor::Variable) && p.rct == Some(6);
            let output = run_pixel_search(
                &image,
                &input,
                &DriverOptions::scout(),
                |params, _, budget| {
                    let size = if cross(params) {
                        150
                    } else if params.predictor == Some(Predictor::Variable) && params.rct == Some(0)
                    {
                        160
                    } else if *params == learned {
                        200
                    } else if params.predictor == Some(Predictor::Zero) {
                        300
                    } else {
                        1000
                    };
                    if budget.beat_bytes.is_some_and(|bound| size >= bound) {
                        return Ok(TrialOutput::Bounded { lower_bound: size });
                    }
                    Ok(TrialOutput::Complete(EncodedOutput {
                        bytes: vec![7; size],
                        artifact: (),
                    }))
                },
                |params, _| {
                    Ok(if cross(params) {
                        75
                    } else if params.predictor == Some(Predictor::Variable) && params.rct == Some(0)
                    {
                        70
                    } else if *params == learned {
                        80
                    } else {
                        1000
                    })
                },
            )
            .unwrap();
            assert_eq!(output.bytes.len(), 150);
            assert!(cross(&output.params));
        }
    }

    #[test]
    fn weak_palette_gain_keeps_search_and_deeper_effort_keeps_witness() {
        use crate::scheduler::search::EncodedOutput;
        use crate::{PixelData, PixelLayout, modular::predictor::Predictor};
        let image = Image::new(
            1024,
            512,
            PixelLayout::Gray,
            PixelData::U8(vec![9; 1024 * 512]),
        )
        .unwrap();
        for (effort, witness_size) in [(9, 299), (10, 200)] {
            let input = CandidateParams::default().backend(effort).unwrap();
            let learned = literal(&input, 9, 0.01).unwrap();
            let output = run_pixel_search(
                &image,
                &input,
                &DriverOptions::scout(),
                |params, _, budget| {
                    let size = if *params == learned {
                        witness_size
                    } else if params.predictor == Some(Predictor::Zero) {
                        300
                    } else {
                        1000
                    };
                    if budget.beat_bytes.is_some_and(|bound| size >= bound) {
                        return Ok(TrialOutput::Bounded { lower_bound: size });
                    }
                    Ok(TrialOutput::Complete(EncodedOutput {
                        bytes: vec![7; size],
                        artifact: (),
                    }))
                },
                |_, _| Ok(100),
            )
            .unwrap();
            assert!(output.report.probe_encodes > 0);
            assert!(output.report.full_encodes > 3);
            assert!(output.bytes.len() <= witness_size);
        }
    }
}
