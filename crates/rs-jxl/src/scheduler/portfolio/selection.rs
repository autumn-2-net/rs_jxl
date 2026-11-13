//! Crop scores are routing heuristics, never whole-image output bounds.
use super::{Candidate, Family, Options, Report, Trial};

/// Compare only measured crops, requiring matching baseline evidence for each.
/// Full confirmation is enforced separately before selecting full-image trials.
pub(super) fn ratio(trial: &Trial, baseline: &Trial) -> Option<f64> {
    if trial.error.is_some()
        || baseline.error.is_some()
        || trial.probe_bytes.len() != baseline.probe_bytes.len()
    {
        return None;
    }
    let mut after = 0u128;
    let mut before = 0u128;
    for (a, b) in trial.probe_bytes.iter().zip(&baseline.probe_bytes) {
        if let Some(a) = a {
            after += *a as u128;
            before += (*b)? as u128;
        }
    }
    (before > 0).then(|| after as f64 / before as f64)
}

fn patch_signal(report: &Report) -> bool {
    report.features.repeated_tile_fraction > 0.0 || report.features.flat_fraction > 0.5
}

fn patch(trial: &Trial) -> bool {
    matches!(trial.candidate.family(), Family::Source | Family::Tile)
}

fn sort_scores(report: &Report, scores: &mut [(usize, f64)]) {
    scores.sort_by(|&(a, ar), &(b, br)| {
        ar.total_cmp(&br)
            .then_with(|| {
                report.candidates[b]
                    .priority
                    .total_cmp(&report.candidates[a].priority)
            })
            .then(a.cmp(&b))
    });
}

/// Rank crop-zero survivors within the remaining all-crop confirmation budget.
/// The caller batches all confirmation crops and drains their results, recording
/// errors without cancelling the remaining crops after an individual failure.
pub(super) fn confirmations(report: &Report, order: &[usize], options: &Options) -> Vec<usize> {
    if options.max_finalists == 0 || report.regions.is_empty() {
        return Vec::new();
    }
    let Some(baseline) = report.candidates.first() else {
        return Vec::new();
    };
    if baseline.candidate != Candidate::Baseline
        || baseline.error.is_some()
        || baseline.probe_bytes.len() != report.regions.len()
        || baseline.probe_bytes.iter().any(Option::is_none)
    {
        return Vec::new();
    }
    let mut scores = Vec::new();
    for &i in order {
        let Some(trial) = report.candidates.get(i) else {
            continue;
        };
        if trial.candidate == Candidate::Baseline
            || trial.probe_bytes.first().copied().flatten().is_none()
            || scores.iter().any(|&(index, _)| index == i)
        {
            continue;
        }
        if let Some(score) = ratio(trial, baseline) {
            scores.push((i, score));
        }
    }
    sort_scores(report, &mut scores);
    let remaining = options
        .max_probe_encodes
        .saturating_sub(report.probe_encodes);
    let extra_crops = report.regions.len() - 1;
    let slots = if extra_crops == 0 {
        scores.len()
    } else {
        remaining / extra_crops
    };
    if slots == 0 {
        return Vec::new();
    }
    // Keep one patch inside the executable prefix, not merely at the tail of
    // a ranking that the caller's crop budget will never reach.
    if patch_signal(report) {
        if let Some(position) = scores
            .iter()
            .position(|&(i, _)| patch(&report.candidates[i]))
        {
            if position >= slots {
                let patch_score = scores[position].1;
                let cutoff_score = scores[slots - 1].1;
                let no_good_scores = scores[0].1 > 1.0 + options.uncertainty;
                // Repetitions justify a strong witness. Flatness alone cannot
                // displace a clearly better crop at the confirmation cutoff.
                if report.features.repeated_tile_fraction > 0.0
                    || patch_score <= cutoff_score + options.uncertainty
                    || no_good_scores
                {
                    let witness = scores.remove(position);
                    scores.insert(slots - 1, witness);
                }
            }
        }
    }
    scores.truncate(slots);
    scores.into_iter().map(|(i, _)| i).collect()
}

fn confident(trial: &Trial, baseline: &Trial, score: f64) -> bool {
    let wins = trial
        .probe_bytes
        .iter()
        .zip(&baseline.probe_bytes)
        .filter(|(a, b)| a < b)
        .count();
    score < 1.0 && wins >= trial.probe_bytes.len().div_ceil(2)
}

pub(super) fn finalists(report: &mut Report, options: &Options) -> Vec<usize> {
    if options.max_finalists == 0 || report.regions.is_empty() {
        return Vec::new();
    }
    let Some(baseline) = report.candidates.first() else {
        return Vec::new();
    };
    if baseline.candidate != Candidate::Baseline
        || baseline.probe_bytes.len() != report.regions.len()
        || baseline.probe_bytes.iter().any(Option::is_none)
        || ratio(baseline, baseline).is_none()
    {
        return Vec::new();
    }
    let mut scores: Vec<_> = report
        .candidates
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, t)| {
            t.candidate != Candidate::Baseline && t.probe_bytes.iter().all(Option::is_some)
        })
        .filter_map(|(i, t)| ratio(t, baseline).map(|r| (i, r)))
        .collect();
    sort_scores(report, &mut scores);
    let reserved = patch_signal(report)
        .then(|| {
            scores
                .iter()
                .find(|&&(i, _)| patch(&report.candidates[i]))
                .copied()
        })
        .flatten()
        .filter(|&(_, patch_score)| {
            report.features.repeated_tile_fraction > 0.0
                || scores
                    .iter()
                    .filter(|&&(i, score)| {
                        !patch(&report.candidates[i])
                            && score < patch_score - options.uncertainty
                            && (confident(&report.candidates[i], baseline, score)
                                || score <= 1.0 + options.uncertainty)
                    })
                    .count()
                    < options.max_finalists
        });
    let mut selected: Vec<usize> = reserved.iter().map(|&(i, _)| i).collect();
    // A reserved patch spends the one speculative slot, even with good crops.
    let mut uncertain_used = reserved.is_some();
    while selected.len() < options.max_finalists {
        let mut eligible: Vec<_> = scores
            .iter()
            .copied()
            .filter(|&(i, r)| {
                !selected.contains(&i)
                    && (confident(&report.candidates[i], baseline, r)
                        || (!uncertain_used && r <= 1.0 + options.uncertainty))
            })
            .collect();
        // Identical full crop vectors within one family are correlated evidence,
        // not proof of semantic equality or identical whole-image output.
        let correlated = |i: usize| {
            selected.iter().any(|&j| {
                report.candidates[j].candidate.family() == report.candidates[i].candidate.family()
                    && report.candidates[j].probe_bytes == report.candidates[i].probe_bytes
            })
        };
        if eligible.iter().any(|&(i, _)| !correlated(i)) {
            eligible.retain(|&(i, _)| !correlated(i));
        }
        let Some(&(best, best_score)) = eligible.first() else {
            break;
        };
        // Diversity is a near-tie preference, never a hard family exclusion.
        let (next, score) = eligible
            .iter()
            .copied()
            .find(|&(i, r)| {
                r <= best_score + options.uncertainty
                    && !selected.iter().any(|&j| {
                        report.candidates[j].candidate.family()
                            == report.candidates[i].candidate.family()
                    })
            })
            .unwrap_or((best, best_score));
        uncertain_used |= !confident(&report.candidates[next], baseline, score);
        selected.push(next);
    }
    // Execute winners by crop score, rather than executing the reserved slot first.
    let mut chosen: Vec<_> = scores
        .iter()
        .copied()
        .filter(|(i, _)| selected.contains(i))
        .collect();
    sort_scores(report, &mut chosen);
    for &(i, score) in &scores {
        let is_confident = confident(&report.candidates[i], &report.candidates[0], score);
        let trial = &mut report.candidates[i];
        if selected.contains(&i) {
            trial.decision = "finalist";
            trial.reason = if reserved.is_some_and(|(index, _)| index == i) {
                if report.features.repeated_tile_fraction > 0.0 {
                    "reserved full-image patch witness; crops can miss distant repetitions"
                } else {
                    "weak flat-image patch witness; stronger crop alternatives retain their slots"
                }
            } else if !is_confident {
                "bounded uncertainty witness; crop score is not conclusive"
            } else {
                "confirmed crop benefit; diversity only breaks near ties"
            }
            .into();
        } else {
            trial.decision = "probe-rejected";
            trial.reason = "crop ranking or work budget; not proof of whole-image loss".into();
        }
    }
    chosen.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{portfolio_features::Features, scan::ScanRegion};

    fn trial(candidate: Candidate, sizes: &[Option<usize>]) -> Trial {
        Trial {
            candidate,
            decision: "queued",
            reason: String::new(),
            priority: 1.0,
            probe_bytes: sizes.to_vec(),
            probe_seconds: 0.0,
            full_bytes: None,
            full_lower_bound: None,
            full_seconds: None,
            error: None,
        }
    }

    fn scored(candidate: Candidate, sizes: &[usize]) -> Trial {
        trial(
            candidate,
            &sizes.iter().copied().map(Some).collect::<Vec<_>>(),
        )
    }

    fn report(candidates: Vec<Trial>) -> Report {
        let crops = candidates.first().map_or(0, |t| t.probe_bytes.len());
        Report {
            features: Features::default(),
            regions: vec![
                ScanRegion {
                    x: 0,
                    y: 0,
                    width: 32,
                    height: 32
                };
                crops
            ],
            candidates,
            winner: Candidate::Baseline,
            seconds: 0.0,
            scout_seconds: 0.0,
            probe_encodes: 0,
            probe_pixels: 0,
            full_encodes: 0,
            preparation_reuse: Default::default(),
            probe_workers: 0,
        }
    }

    #[test]
    fn news_like_tiny_crop_gains_do_not_displace_source() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[1982, 106, 101]),
            scored(Candidate::OffsetDistribution, &[1980, 106, 97]),
            scored(Candidate::Source, &[1982, 106, 101]),
            scored(Candidate::Prev1, &[1981, 105, 100]),
        ]);
        r.features.repeated_tile_fraction = 0.1;
        assert_eq!(finalists(&mut r, &Options::default()), vec![1, 2]);
    }

    #[test]
    fn flat_only_16bit_patch_does_not_displace_two_clear_crop_winners() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[4899, 4939, 4265]),
            scored(Candidate::ShallowFull, &[3809, 3731, 3937]),
            scored(Candidate::Prev1Distribution, &[4782, 4868, 4154]),
            scored(Candidate::Source, &[4899, 4939, 4265]),
        ]);
        r.features.flat_fraction = 0.868;
        assert_eq!(finalists(&mut r, &Options::default()), vec![1, 2]);
        assert_eq!(r.candidates[3].decision, "probe-rejected");

        for trial in r.candidates.iter_mut().skip(1) {
            trial.probe_bytes[1..].fill(None);
        }
        r.probe_encodes = 6;
        let options = Options {
            max_probe_encodes: 10,
            ..Default::default()
        };
        assert_eq!(confirmations(&r, &[1, 2, 3], &options), vec![1, 2]);
        r.features.repeated_tile_fraction = 0.1;
        assert_eq!(confirmations(&r, &[1, 2, 3], &options), vec![1, 3]);
    }

    #[test]
    fn correlated_16bit_shallow_scores_defer_to_novel_cross_channel_evidence() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[4899, 4939, 4265]),
            scored(Candidate::ShallowFull, &[3809, 3731, 3937]),
            scored(Candidate::ShallowFullPrev1, &[3809, 3731, 3937]),
            scored(Candidate::Prev1Distribution, &[4782, 4868, 4154]),
            scored(Candidate::Source, &[4899, 4939, 4265]),
        ]);
        r.features.flat_fraction = 0.868;
        assert_eq!(finalists(&mut r, &Options::default()), vec![1, 3]);
        assert_eq!(r.candidates[2].decision, "probe-rejected");
        assert_eq!(r.candidates[4].decision, "probe-rejected");
    }

    #[test]
    fn correlated_candidates_can_fill_slots_without_novel_eligible_alternatives() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            scored(Candidate::ShallowFull, &[70; 3]),
            scored(Candidate::ShallowFullPrev1, &[70; 3]),
            scored(Candidate::OffsetDistribution, &[150; 3]),
        ]);
        assert_eq!(finalists(&mut r, &Options::default()), vec![1, 2]);
    }

    #[test]
    fn same_family_equal_ratios_with_distinct_vectors_are_novel() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            scored(Candidate::Prev1, &[70, 75, 80]),
            scored(Candidate::Prev2, &[75, 70, 80]),
            scored(Candidate::OffsetDistribution, &[95; 3]),
        ]);
        assert_eq!(finalists(&mut r, &Options::default()), vec![1, 2]);
    }

    #[test]
    fn weak_flat_patch_can_coexist_with_one_clear_winner_but_not_replace_it() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            scored(Candidate::Prev1, &[80; 3]),
            scored(Candidate::Source, &[100; 3]),
        ]);
        r.features.flat_fraction = 0.6;
        assert_eq!(finalists(&mut r, &Options::default()), vec![1, 2]);
        let options = Options {
            max_finalists: 1,
            ..Default::default()
        };
        assert_eq!(finalists(&mut r, &options), vec![1]);
    }

    #[test]
    fn weak_flat_patch_protects_near_ties_in_finalists_and_confirmation() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[1000; 3]),
            scored(Candidate::Prev1, &[995; 3]),
            scored(Candidate::OffsetDistribution, &[996; 3]),
            scored(Candidate::Source, &[1000; 3]),
        ]);
        r.features.flat_fraction = 0.6;
        assert_eq!(finalists(&mut r, &Options::default()), vec![1, 3]);
        for trial in r.candidates.iter_mut().skip(1) {
            trial.probe_bytes[1..].fill(None);
        }
        let options = Options {
            max_probe_encodes: 4,
            ..Default::default()
        };
        assert_eq!(confirmations(&r, &[1, 2, 3], &options), vec![1, 3]);
    }

    #[test]
    fn weak_flat_confirmation_still_reserves_patch_without_good_crop_scores() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            trial(Candidate::Prev1, &[Some(110), None, None]),
            trial(Candidate::OffsetDistribution, &[Some(120), None, None]),
            trial(Candidate::Source, &[Some(150), None, None]),
        ]);
        r.features.flat_fraction = 0.6;
        let options = Options {
            max_probe_encodes: 4,
            ..Default::default()
        };
        assert_eq!(confirmations(&r, &[1, 2, 3], &options), vec![1, 3]);
    }

    #[test]
    fn two_strong_cross_channel_candidates_are_allowed() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            scored(Candidate::Prev1, &[70; 3]),
            scored(Candidate::Prev2, &[75; 3]),
            scored(Candidate::OffsetDistribution, &[95; 3]),
        ]);
        assert_eq!(finalists(&mut r, &Options::default()), vec![1, 2]);
    }

    #[test]
    fn diversity_only_displaces_near_ties() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            scored(Candidate::Prev1, &[70; 3]),
            scored(Candidate::Prev2, &[75; 3]),
            scored(Candidate::OffsetDistribution, &[76; 3]),
        ]);
        assert_eq!(finalists(&mut r, &Options::default()), vec![1, 3]);
    }

    #[test]
    fn confirmation_reserves_patch_and_priority_breaks_ties() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            trial(Candidate::Prev1, &[Some(90), None, None]),
            trial(Candidate::Prev2, &[Some(91), None, None]),
            trial(Candidate::Source, &[Some(100), None, None]),
            trial(Candidate::SourceParallel, &[Some(100), None, None]),
        ]);
        r.candidates[4].priority = 1.05;
        r.features.flat_fraction = 0.6;
        r.features.repeated_tile_fraction = 0.1;
        r.probe_encodes = 7;
        let options = Options {
            max_probe_encodes: 11,
            ..Default::default()
        };
        assert_eq!(confirmations(&r, &[1, 2, 3, 4], &options), vec![1, 4]);
        assert!(finalists(&mut r, &options).is_empty());
        let options = Options {
            max_probe_encodes: 9,
            ..options
        };
        assert_eq!(confirmations(&r, &[1, 2, 3, 4], &options), vec![4]);
    }

    #[test]
    fn patch_reservation_spends_the_only_uncertainty_witness() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            scored(Candidate::Source, &[150; 3]),
            scored(Candidate::Tile16, &[160; 3]),
            scored(Candidate::OffsetDistribution, &[100; 3]),
        ]);
        r.features.flat_fraction = 0.6;
        let options = Options {
            max_finalists: 4,
            ..Default::default()
        };
        assert_eq!(finalists(&mut r, &options), vec![1]);
    }

    #[test]
    fn only_one_general_uncertainty_witness_and_parallel_source_wins_ties() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            scored(Candidate::Source, &[100; 3]),
            scored(Candidate::SourceParallel, &[100; 3]),
            scored(Candidate::OffsetDistribution, &[100; 3]),
        ]);
        r.candidates[2].priority = 1.05;
        assert_eq!(finalists(&mut r, &Options::default()), vec![2]);
    }

    #[test]
    fn errors_incomplete_empty_and_zero_denominators_are_excluded() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            scored(Candidate::Prev1, &[1; 3]),
            trial(Candidate::Prev2, &[Some(1), None, None]),
            trial(Candidate::Source, &[]),
            trial(Candidate::Tile16, &[None; 3]),
        ]);
        r.candidates[1].error = Some("failed probe".into());
        assert!(finalists(&mut r, &Options::default()).is_empty());
        assert_eq!(
            confirmations(&r, &[1, 2, 3, 4], &Options::default()),
            vec![2]
        );
        assert!(ratio(&r.candidates[1], &r.candidates[0]).is_none());
        assert!(
            ratio(
                &scored(Candidate::Prev1, &[1]),
                &scored(Candidate::Baseline, &[0])
            )
            .is_none()
        );
        assert!(
            ratio(
                &scored(Candidate::Prev1, &[1]),
                &trial(Candidate::Baseline, &[None])
            )
            .is_none()
        );
        assert!(finalists(&mut report(Vec::new()), &Options::default()).is_empty());
        assert!(
            finalists(
                &mut report(vec![trial(Candidate::Baseline, &[])]),
                &Options::default()
            )
            .is_empty()
        );
        r.candidates[0].probe_bytes[1] = None;
        assert!(finalists(&mut r, &Options::default()).is_empty());
        assert!(confirmations(&r, &[2], &Options::default()).is_empty());
    }

    #[test]
    fn zero_budgets_and_single_finalist_reservation_are_respected() {
        let mut r = report(vec![
            scored(Candidate::Baseline, &[100; 3]),
            scored(Candidate::Prev1, &[80; 3]),
            scored(Candidate::Source, &[100; 3]),
        ]);
        r.features.repeated_tile_fraction = 0.1;
        let options = Options {
            max_finalists: 0,
            ..Default::default()
        };
        assert!(finalists(&mut r, &options).is_empty());
        assert!(confirmations(&r, &[1, 2], &options).is_empty());
        let options = Options {
            max_probe_encodes: 0,
            ..Default::default()
        };
        assert!(confirmations(&r, &[1, 2], &options).is_empty());
        let options = Options {
            max_finalists: 1,
            ..Default::default()
        };
        assert_eq!(finalists(&mut r, &options), vec![2]);
    }
}
