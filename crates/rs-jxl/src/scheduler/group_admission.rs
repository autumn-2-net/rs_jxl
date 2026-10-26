//! Cheap-front admission only. Rejection is heuristic, never an encoded-byte bound.
use super::{Candidate, DriverOptions, LosslessScan, Predictor};

/// Return a mask in plan order without changing candidates or performing group work.
/// Natural and Zero representations remain available for real confirmation.
pub(super) fn admission(
    plan: &[Candidate],
    ranked: &[(Candidate, Vec<usize>)],
    driver: &DriverOptions,
    scan: &LosslessScan,
) -> Vec<bool> {
    let admit_all = || vec![true; plan.len()];
    let Some((leader, leader_scores)) = ranked.first() else {
        return admit_all();
    };
    if leader_scores.len() < 3
        || ranked
            .iter()
            .enumerate()
            .any(|(index, (candidate, scores))| {
                scores.len() != leader_scores.len()
                    || scores.contains(&0)
                    || scores
                        .iter()
                        .try_fold(0usize, |sum, &score| sum.checked_add(score))
                        .is_none()
                    || !plan.iter().any(|c| c.params == candidate.params)
                    || ranked[..index]
                        .iter()
                        .any(|(c, _)| c.params == candidate.params)
            })
    {
        return admit_all();
    }
    let near_tie_consensus = driver.scout_near_tie_consensus && scan.likely_continuous_tone();
    plan.iter()
        .map(|candidate| {
            if candidate.params.predictor.is_none()
                || candidate.params.predictor == Some(Predictor::Zero)
                || candidate.params == leader.params
            {
                return true;
            }
            let Some((_, scores)) = ranked.iter().find(|(c, _)| c.params == candidate.params)
            else {
                // A valid front may deliberately defer or reject Variable RCTs.
                // Unknown representations absent from that front remain protected.
                return candidate.params.predictor != Some(Predictor::Variable);
            };
            super::super::driver::scout_pilot::pilot_eligible(
                9,
                driver.scout_pilot_consensus,
                near_tie_consensus,
                &leader.params,
                leader_scores,
                &candidate.params,
                scores,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::{CandidateParams, literal, winner_neighborhood};
    use super::*;

    fn fixture() -> (Vec<Candidate>, DriverOptions, LosslessScan) {
        let auto = CandidateParams::default().backend(9).unwrap();
        let mut plan = vec![
            Candidate::new("automatic", auto.clone()),
            Candidate::new("neighborhood", winner_neighborhood(&auto, 3)),
            Candidate::new("literal", literal(&auto, 9, 0.01).unwrap()),
        ];
        for (name, rct) in [
            ("raw-cross-channel", 0),
            ("ycocg-cross-channel", 6),
            ("rct10-cross-channel", 10),
            ("rct17-cross-channel", 17),
            ("rct26-cross-channel", 26),
        ] {
            plan.push(Candidate::new(
                name,
                CandidateParams {
                    predictor: Some(Predictor::Variable),
                    rct: Some(rct),
                    ..auto.clone()
                },
            ));
        }
        let scan = LosslessScan {
            sampled_pixels: 1024,
            sampled_colors: 1000,
            ..Default::default()
        };
        (plan, DriverOptions::scout(), scan)
    }

    #[test]
    fn rank_inversion_keeps_both_natural_candidates_even_when_absent() {
        let (plan, driver, scan) = fixture();
        for include_natural in [false, true] {
            let mut ranked = vec![(plan[3].clone(), vec![100; 3])];
            if include_natural {
                ranked.extend([
                    (plan[0].clone(), vec![200; 3]),
                    (plan[1].clone(), vec![300; 3]),
                ]);
            }
            assert_eq!(
                admission(&plan, &ranked, &driver, &scan),
                [true, true, true, true, false, false, false, false]
            );
        }
    }

    #[test]
    fn rejected_and_deferred_extended_candidates_do_not_reenter() {
        let (plan, driver, scan) = fixture();
        let ranked = vec![
            (plan[0].clone(), vec![100; 3]),
            (plan[3].clone(), vec![102; 3]),
            (plan[4].clone(), vec![200; 3]),
            (plan[5].clone(), vec![200; 3]),
        ];
        assert_eq!(
            admission(&plan, &ranked, &driver, &scan),
            [true, true, true, true, false, false, false, false]
        );
    }

    #[test]
    fn cheap_winner_is_kept_for_every_family() {
        let (mut plan, mut driver, scan) = fixture();
        plan.push(Candidate::new(
            "weighted",
            CandidateParams {
                predictor: Some(Predictor::Weighted),
                ..plan[0].params.clone()
            },
        ));
        driver.scout_pilot_consensus = true;
        driver.scout_near_tie_consensus = true;
        for (index, candidate) in plan.iter().enumerate() {
            let ranked = vec![(candidate.clone(), vec![100; 3])];
            assert!(admission(&plan, &ranked, &driver, &scan)[index]);
        }
    }

    #[test]
    fn matching_uses_parameters_not_names_or_cached_costs() {
        let (plan, driver, scan) = fixture();
        let mut leader = plan[3].clone();
        leader.name = "automatic";
        leader.context_bytes = Some(300);
        leader.bytes = Some(999);
        let ranked = vec![(leader, vec![100; 3])];
        assert_eq!(
            admission(&plan, &ranked, &driver, &scan),
            [true, true, true, true, false, false, false, false]
        );
    }

    #[test]
    fn material_regional_winner_survives_an_aggregate_loss() {
        let (plan, mut driver, scan) = fixture();
        driver.scout_pilot_consensus = true;
        driver.scout_near_tie_consensus = true;
        let ranked = vec![
            (plan[0].clone(), vec![1000; 3]),
            (plan[4].clone(), vec![1150; 3]),
            (plan[3].clone(), vec![800, 1500, 1500]),
        ];
        let mask = admission(&plan, &ranked, &driver, &scan);
        assert!(mask[3]);
        assert!(!mask[4]);
    }

    #[test]
    fn static_leader_and_all_zero_representations_remain_protected() {
        let (mut plan, mut driver, scan) = fixture();
        plan.push(Candidate::new(
            "literal-static",
            literal(&plan[0].params, 9, 0.0).unwrap(),
        ));
        plan.push(Candidate::new(
            "raw-zero",
            CandidateParams {
                predictor: Some(Predictor::Zero),
                palette_colors: 0,
                rct: Some(6),
                ..plan[0].params.clone()
            },
        ));
        for witnesses in [false, true] {
            driver.scout_representation_witness = witnesses;
            for ranked in [
                vec![
                    (plan[8].clone(), vec![100; 3]),
                    (plan[2].clone(), vec![200; 3]),
                ],
                vec![
                    (plan[0].clone(), vec![100; 3]),
                    (plan[8].clone(), vec![200; 3]),
                ],
            ] {
                let mask = admission(&plan, &ranked, &driver, &scan);
                for index in [2, 8, 9] {
                    assert!(mask[index]);
                }
            }
        }
    }

    #[test]
    fn consensus_flags_and_continuous_tone_guard_are_preserved() {
        let (plan, mut driver, scan) = fixture();
        let mut ranked = vec![
            (plan[0].clone(), vec![1000; 3]),
            (plan[3].clone(), vec![1005; 3]),
        ];
        assert!(admission(&plan, &ranked, &driver, &scan)[3]);
        driver.scout_pilot_consensus = true;
        assert!(!admission(&plan, &ranked, &driver, &scan)[3]);
        ranked[1].1 = vec![995, 1020, 1020];
        assert!(admission(&plan, &ranked, &driver, &scan)[3]);
        driver.scout_near_tie_consensus = true;
        assert!(!admission(&plan, &ranked, &driver, &scan)[3]);
        assert!(admission(&plan, &ranked, &driver, &LosslessScan::default())[3]);
    }

    #[test]
    fn missing_empty_and_malformed_scores_admit_the_whole_plan() {
        let (plan, driver, scan) = fixture();
        let all = vec![true; plan.len()];
        assert_eq!(admission(&plan, &[], &driver, &scan), all);
        for scores in [
            vec![],
            vec![100],
            vec![100; 2],
            vec![0; 3],
            vec![usize::MAX; 3],
        ] {
            let ranked = vec![(plan[0].clone(), scores)];
            assert_eq!(admission(&plan, &ranked, &driver, &scan), all);
        }
        for scores in [
            vec![],
            vec![200; 2],
            vec![200; 4],
            vec![200, 0, 200],
            vec![usize::MAX; 3],
        ] {
            let ranked = vec![(plan[0].clone(), vec![100; 3]), (plan[2].clone(), scores)];
            assert_eq!(admission(&plan, &ranked, &driver, &scan), all);
        }
    }

    #[test]
    fn foreign_or_duplicate_front_entries_admit_the_whole_plan() {
        let (plan, driver, scan) = fixture();
        let mut foreign = plan[3].clone();
        foreign.params.rct = Some(40);
        for ranked in [
            vec![(foreign, vec![100; 3])],
            vec![
                (plan[0].clone(), vec![100; 3]),
                (plan[0].clone(), vec![200; 3]),
            ],
        ] {
            assert_eq!(
                admission(&plan, &ranked, &driver, &scan),
                vec![true; plan.len()]
            );
        }
    }

    #[test]
    fn absent_unrecognized_predictor_and_empty_plan_are_safe() {
        let (mut plan, driver, scan) = fixture();
        plan.push(Candidate::new(
            "weighted",
            CandidateParams {
                predictor: Some(Predictor::Weighted),
                ..plan[0].params.clone()
            },
        ));
        let ranked = vec![(plan[0].clone(), vec![100; 3])];
        assert!(admission(&plan, &ranked, &driver, &scan)[8]);
        assert!(admission(&[], &ranked, &driver, &scan).is_empty());
    }
}
