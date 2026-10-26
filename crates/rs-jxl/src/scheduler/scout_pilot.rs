//! Opt-in pilot-first search. Sample estimates choose work, never final bytes.
use super::*;
use crate::modular::predictor::Predictor;

pub(crate) fn competitive(candidate: &[usize], leader: &[usize]) -> bool {
    if candidate.len() != leader.len() || candidate.is_empty() {
        return true;
    }
    let a: f64 = candidate.iter().map(|&n| n as f64).sum();
    let b: f64 = leader.iter().map(|&n| n as f64).sum();
    b == 0.0
        || a <= b * 1.025
        || candidate
            .iter()
            .zip(leader)
            .any(|(&x, &y)| y as f64 >= b * 0.10 && (x as f64) < y as f64 * 0.95)
}

fn consistently_worse(candidate: &[usize], leader: &[usize]) -> bool {
    candidate.len() >= 3
        && candidate.len() == leader.len()
        && candidate
            .iter()
            .zip(leader)
            .all(|(&a, &b)| b != 0 && a as u128 * 400 >= b as u128 * 401)
        && candidate.iter().map(|&n| n as u128).sum::<u128>() * 200
            >= leader.iter().map(|&n| n as u128).sum::<u128>() * 201
}

fn half_percent_worse(candidate: u128, leader: u128) -> bool {
    // Exactly candidate * 200 >= leader * 201 without overflowing either product.
    candidate >= leader && candidate - leader >= leader.div_ceil(200)
}

fn near_tie_worse(candidate: &[usize], leader: &[usize]) -> bool {
    if candidate.len() < 3
        || candidate.len() != leader.len()
        || !candidate
            .iter()
            .zip(leader)
            .all(|(&a, &b)| a != 0 && b != 0 && a as u128 * 100 >= b as u128 * 99)
    {
        return false;
    }
    candidate
        .iter()
        .zip(leader)
        .try_fold((0u128, 0u128), |(a, b), (&x, &y)| {
            Some((a.checked_add(x as u128)?, b.checked_add(y as u128)?))
        })
        .is_some_and(|(a, b)| half_percent_worse(a, b))
}

fn strong_overlap_challenger(candidate: &[usize], automatic: &[usize]) -> bool {
    if candidate.len() < 3
        || candidate.len() != automatic.len()
        || candidate.contains(&0)
        || automatic.contains(&0)
    {
        return true;
    }
    let totals = candidate
        .iter()
        .zip(automatic)
        .try_fold((0u128, 0u128), |(a, b), (&x, &y)| {
            Some((a.checked_add(x as u128)?, b.checked_add(y as u128)?))
        });
    let Some((a, b)) = totals else { return true };
    // The 2% margin is empirical screening, not a 0.25% final-size guarantee
    // or an encoded-byte bound. Two material regional wins preserve alternatives.
    (a <= b && b - a >= b.div_ceil(50))
        || candidate
            .iter()
            .zip(automatic)
            .filter(|&(&x, &y)| y as u128 * 10 >= b && x as u128 * 20 <= y as u128 * 19)
            .count()
            >= 2
}

fn prune_weak_overlap_challengers(
    ranked: &mut Vec<(Candidate, Vec<usize>)>,
    automatic: &CandidateParams,
) -> usize {
    let Some(scores) = ranked
        .iter()
        .find(|(c, _)| c.params == *automatic)
        .map(|(_, scores)| scores.clone())
    else {
        return 0;
    };
    let before = ranked.len();
    ranked.retain(|(candidate, candidate_scores)| {
        candidate.params == *automatic
            || candidate.params.predictor == Some(Predictor::Zero)
            || strong_overlap_challenger(candidate_scores, &scores)
    });
    before - ranked.len()
}

fn coarse_dominated(candidate: &[usize], leader: &[usize]) -> bool {
    candidate.len() >= 3
        && candidate.len() == leader.len()
        && candidate
            .iter()
            .zip(leader)
            .all(|(&a, &b)| b != 0 && a as u128 * 100 >= b as u128 * 101)
}

pub(crate) fn pilot_eligible(
    level: u8,
    consensus: bool,
    near_tie_consensus: bool,
    leader_params: &CandidateParams,
    leader_scores: &[usize],
    candidate: &CandidateParams,
    scores: &[usize],
) -> bool {
    // Preserve the learned full-image witness when tiny palette probes prefer
    // its static sibling; both candidates retain the same backend effort.
    let learned_palette_witness = leader_params.predictor == Some(Predictor::Zero)
        && leader_params.learning_fraction == 0.0
        && candidate.predictor == Some(Predictor::Zero)
        && candidate.learning_fraction > 0.0;
    let consensus_reject = level == 9
        && candidate.predictor != Some(Predictor::Zero)
        && leader_params.predictor.is_none()
        && ((consensus && consistently_worse(scores, leader_scores))
            || (near_tie_consensus && near_tie_worse(scores, leader_scores)));
    let rank_reject = level == 9 && !learned_palette_witness && !competitive(scores, leader_scores);
    !(consensus_reject || rank_reject)
}

fn remove_native_rct_aliases(plan: &mut Vec<Candidate>, channels: usize) -> usize {
    if channels >= 3 {
        return 0;
    }
    let raw: Vec<_> = plan
        .iter()
        .filter(|c| c.params.predictor == Some(Predictor::Variable) && c.params.rct == Some(0))
        .map(|c| c.params.clone())
        .collect();
    let before = plan.len();
    plan.retain(|candidate| {
        if candidate.params.predictor != Some(Predictor::Variable)
            || candidate.params.rct != Some(6)
        {
            return true;
        }
        let mut comparable = candidate.params.clone();
        comparable.rct = Some(0);
        !raw.contains(&comparable)
    });
    before - plan.len()
}

/// Shared by serial pilot and the thread-local overlap front. Only complete,
/// single-scale score vectors leave this function.
pub(crate) fn rank_front<F, G>(
    mut plan: Vec<Candidate>,
    level: u8,
    options: &DriverOptions,
    scan: Option<&LosslessScan>,
    screen: &mut CandidateScreen<'_, F>,
    context_screen: &mut CandidateScreen<'_, G>,
    events: &mut Vec<SearchEvent>,
) -> Result<Vec<(Candidate, Vec<usize>)>>
where
    F: FnMut(&CandidateParams, &Image) -> Result<usize>,
    G: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    // Defer additional transforms until a cross-channel representation is
    // competitive in the initial front. Crop winners can reverse at full size.
    let mut extended = Vec::new();
    if options.scout_representation_witness && options.plan.extended_rct {
        plan.retain(|candidate| {
            let defer = candidate.params.predictor == Some(Predictor::Variable)
                && matches!(candidate.params.rct, Some(10 | 17 | 26));
            if defer {
                extended.push(candidate.clone());
            }
            !defer
        });
    }
    let mut coarse_ranked = None;
    if level == 9
        && options.scout_staged_probes
        && scan.is_some_and(|scan| scan.likely_continuous_tone() && !scan.literal_candidate)
    {
        let mut coarse = context_screen.score_plan(plan)?;
        if crate::profile::enabled() {
            for (candidate, scores) in &coarse {
                crate::profile::note(
                    "scheduler.coarse",
                    format_args!("family={} scores={scores:?}", candidate.name),
                );
            }
        }
        context_screen.release_samples_and_cache();
        coarse.sort_by_key(|(c, _)| c.context_bytes);
        let leader = coarse[0].1.clone();
        let decisive = options.scout_coarse_consensus
            && coarse[0].0.params.predictor.is_none()
            && coarse.len() > 1
            && coarse[1..]
                .iter()
                .all(|(_, scores)| coarse_dominated(scores, &leader));
        if decisive {
            coarse_ranked = Some(vec![coarse.remove(0)]);
            plan = Vec::new();
            events.push(SearchEvent::Route {
                name: "adaptive-scout-coarse-consensus",
                candidates: 1,
            });
        } else {
            plan = coarse
                .into_iter()
                .filter(|(c, scores)| {
                    // Preserve both natural representations despite crop model costs.
                    c.params.predictor.is_none()
                        || (competitive(scores, &leader)
                            && !(options.scout_coarse_consensus
                                && coarse_dominated(scores, &leader)))
                })
                .map(|(c, _)| c)
                .collect();
        }
    }
    let mut ranked = match coarse_ranked {
        Some(ranked) => ranked,
        None => screen.score_plan(plan)?,
    };
    ranked.sort_by_key(|(c, _)| c.context_bytes);
    if !extended.is_empty()
        && ranked.first().is_some_and(|(_, leader_scores)| {
            ranked.iter().any(|(candidate, scores)| {
                candidate.params.predictor == Some(Predictor::Variable)
                    && competitive(scores, leader_scores)
            })
        })
    {
        ranked.extend(screen.score_plan(extended)?);
        ranked.sort_by_key(|(c, _)| c.context_bytes);
        events.push(SearchEvent::Route {
            name: "adaptive-scout-extended-cross",
            candidates: ranked.len(),
        });
    }
    events.push(SearchEvent::Frontier {
        candidates: ranked.iter().map(|(c, _)| c.clone()).collect(),
    });
    if crate::profile::enabled() {
        for (c, scores) in &ranked {
            crate::profile::note(
                "scheduler.pilot",
                format_args!("family={} scores={scores:?}", c.name),
            );
        }
    }
    Ok(ranked)
}

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    pub(super) fn run_scout_pilot(&mut self) -> Result<bool> {
        let automatic = self.input.backend(9)?;
        // Flat/low-color images can reverse a small crop margin at full size.
        // A scan-less custom driver keeps its explicit opt-in behavior.
        let near_tie_consensus = self.options.scout_near_tie_consensus
            && self.scan.is_none_or(LosslessScan::likely_continuous_tone);
        let mut plan = vec![
            Candidate::new("automatic", automatic.clone()),
            Candidate::new(
                "neighborhood",
                winner_neighborhood(&automatic, self.image.channel_count()),
            ),
            Candidate::new("literal", literal(self.input, 9, 0.01)?),
        ];
        plan.extend(
            lossless_probe_plan(self.input, self.options.plan)
                .into_iter()
                .filter(|c| c.params.predictor == Some(Predictor::Variable)),
        );
        // Large flat palettes can spend more on a learned tree than on their
        // residuals. Compare a fixed-tree tool at the SAME backend effort.
        if self.scan.is_some_and(|scan| {
            scan.sampled_pixels != 0
                && scan.sampled_colors <= 256
                && scan.sampled_flat * 4 >= scan.sampled_pixels * 3
        }) {
            plan.push(Candidate::new(
                "literal-static",
                literal(self.input, 9, 0.0)?,
            ));
        }
        if self.options.scout_native_rct_aliases {
            let aliases = remove_native_rct_aliases(&mut plan, self.image.channel_count());
            if aliases != 0 {
                self.events.push(SearchEvent::Route {
                    name: "adaptive-scout-native-rct-aliases",
                    candidates: aliases,
                });
            }
        }
        let (mut ranked, overlapped) = match self.overlap_front(&automatic, &plan)? {
            Some(ranked) => (ranked, true),
            None => (
                rank_front(
                    plan,
                    self.level,
                    self.options,
                    self.scan,
                    &mut self.screen,
                    &mut self.context_screen,
                    &mut self.events,
                )?,
                false,
            ),
        };
        let completed_automatic = overlapped
            && self.search.find(&automatic).is_some_and(|index| {
                let trial = &self.search.trials()[index];
                trial.baseline && trial.bytes.is_some()
            });
        if completed_automatic && self.level == 9 && near_tie_consensus {
            let removed = prune_weak_overlap_challengers(&mut ranked, &automatic);
            if removed != 0 {
                self.events.push(SearchEvent::Route {
                    name: "adaptive-scout-weak-overlap-challengers",
                    candidates: removed,
                });
            }
        }
        let (leader, mut leader_scores) = ranked.remove(0);
        // Crops cannot certify full-frame palette/patch behavior. Retain a real
        // output for those distinct representations before pruning by scores.
        let witness = ranked.iter().find(|(candidate, scores)| {
            candidate.params.predictor == Some(Predictor::Zero)
                && competitive(scores, &leader_scores)
        });
        self.pilot_representation_witness(if leader.params.predictor == Some(Predictor::Zero) {
            &leader.params
        } else {
            witness.map_or(&leader.params, |(candidate, _)| &candidate.params)
        })?;
        // The first real candidate must complete. Subsequent candidates may
        // stop only at a proven encoded-byte bound, never a probe estimate.
        let mut leader_params = leader.params.clone();
        let mut selected = 1;
        if overlapped {
            // This completed output cannot improve the incumbent again. Keep
            // its real trial, but do not spend a finalist slot on its reuse.
            ranked.retain(|(candidate, _)| candidate.params != automatic);
        }
        let overlap_finalists = completed_automatic && self.search.find(&leader_params).is_none();
        if overlap_finalists {
            // Auto has already completed. Submit the original rank leader and
            // its consecutive eligible peers through the bounded finalist loop.
            ranked.insert(0, (leader, leader_scores.clone()));
        } else {
            let initial_width = self.initial_wave_capacity()?;
            let mut initial_params = vec![leader_params.clone()];
            for (candidate, scores) in ranked.iter().take(initial_width - 1) {
                if !pilot_eligible(
                    self.level,
                    self.options.scout_pilot_consensus,
                    near_tie_consensus,
                    &leader_params,
                    &leader_scores,
                    &candidate.params,
                    scores,
                ) {
                    break;
                }
                initial_params.push(candidate.params.clone());
            }
            if let Some(sizes) = self.full_initial_wave(9, &initial_params)? {
                selected = sizes.len();
                let mut best = sizes[0];
                for ((candidate, scores), size) in ranked
                    .drain(..sizes.len() - 1)
                    .zip(sizes.into_iter().skip(1))
                {
                    if size < best {
                        best = size;
                        leader_params = candidate.params;
                        leader_scores = scores;
                    }
                }
            } else {
                let baseline = self.search.trials().is_empty();
                self.full(9, leader_params.clone(), baseline, false)?;
            }
        }
        let batch_size = if overlap_finalists {
            self.finalist_wave_capacity()?
        } else if self.options.scout_parallel_finalists {
            2
        } else {
            1
        };
        let mut remaining = ranked.into_iter().peekable();
        loop {
            let mut pending = Vec::with_capacity(batch_size);
            while let Some((candidate, scores)) = remaining.peek() {
                if !pilot_eligible(
                    self.level,
                    self.options.scout_pilot_consensus,
                    near_tie_consensus,
                    &leader_params,
                    &leader_scores,
                    &candidate.params,
                    scores,
                ) {
                    // A pending winner may change the comparison scale. Do not
                    // consume rejected entries before that wave is committed.
                    if !pending.is_empty() {
                        break;
                    }
                    remaining.next();
                    continue;
                }
                pending.push(remaining.next().expect("peeked finalist"));
                if pending.len() == batch_size {
                    break;
                }
            }
            if pending.is_empty() {
                break;
            }
            let mut before = self.best_size(10)?;
            let params: Vec<_> = pending.iter().map(|(c, _)| c.params.clone()).collect();
            let sizes = self.full_finalists(9, &params)?;
            for ((candidate, scores), size) in pending.into_iter().zip(sizes) {
                selected += 1;
                if let Some(n) = size
                    && n < before
                {
                    before = n;
                    leader_scores = scores;
                    leader_params = candidate.params;
                }
            }
        }
        if self.level == 10 {
            let winner = self.best_params(10)?;
            for params in [winner.backend(10)?, self.input.backend(10)?] {
                if self.search.find(&params).is_none() {
                    self.full(10, params, false, false)?;
                    selected += 1;
                }
            }
        }
        self.events.push(SearchEvent::Route {
            name: "adaptive-scout-pilot-first",
            candidates: selected,
        });
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout, scheduler::search::EncodedOutput};

    pub(super) fn fixture() -> (Image, DriverOptions) {
        let image = Image::new(
            1024,
            512,
            PixelLayout::Gray,
            PixelData::U8(vec![0; 1024 * 512]),
        )
        .unwrap();
        let mut options = DriverOptions::scout();
        options.scout_pilot_first = true;
        options.coverage_scan = false;
        (image, options)
    }

    #[test]
    fn native_rct_aliases_are_opt_in_and_require_pilot_first() {
        assert!(!DriverOptions::default().scout_native_rct_aliases);
        assert!(!DriverOptions::scout().scout_native_rct_aliases);
        let (image, mut options) = fixture();
        options.scout_pilot_first = false;
        options.scout_native_rct_aliases = true;
        let error = run_pixel_search::<_, _, ()>(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |_, _, _| panic!("invalid alias policy must not encode"),
            |_, _| panic!("invalid alias policy must not probe"),
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput(message)
            if message == "pilot refinements require pilot-first"));
    }

    #[test]
    fn extended_cross_is_scored_for_competitive_not_only_leading_cross_channels() {
        for cross_score in [50, 101, 300] {
            let (image, mut options) = fixture();
            options.scout_representation_witness = true;
            options.plan.extended_rct = true;
            let mut probed = Vec::new();
            let result = run_pixel_search(
                &image,
                &CandidateParams::default().backend(9).unwrap(),
                &options,
                |params, _, budget| {
                    let size = if params.rct == Some(17) { 40 } else { 100 };
                    if budget.beat_bytes.is_some_and(|bound| size >= bound) {
                        Ok(TrialOutput::Bounded { lower_bound: size })
                    } else {
                        Ok(TrialOutput::Complete(EncodedOutput {
                            bytes: vec![17; size],
                            artifact: params.rct,
                        }))
                    }
                },
                |params, _| {
                    probed.push(params.rct);
                    Ok(if params.rct == Some(17) {
                        5
                    } else if params.predictor == Some(Predictor::Variable) {
                        cross_score
                    } else {
                        100
                    })
                },
            )
            .unwrap();
            for rct in [10, 17, 26] {
                assert_eq!(probed.contains(&Some(rct)), cross_score <= 101);
            }
            assert_eq!(
                result.bytes.len(),
                if cross_score <= 101 { 40 } else { 100 }
            );
            assert_eq!(result.params.rct == Some(17), cross_score <= 101);
        }
    }

    #[test]
    fn native_rct_aliases_preserve_channel_boundaries_work_limits_and_routes() {
        use crate::metadata::{ExtraChannelInfo, ExtraChannelType};

        let (gray, options) = fixture();
        let width = gray.width();
        let height = gray.height();
        let pixels = width as usize * height as usize;
        let mut metadata = gray.metadata().clone();
        metadata.extra_channels =
            vec![ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap(); 2];
        let gray_extras = gray
            .clone()
            .with_metadata(metadata, vec![PixelData::U8(vec![0; pixels]); 2])
            .unwrap();
        let cases = [
            (gray, true),
            (
                Image::new(
                    width,
                    height,
                    PixelLayout::GrayAlpha,
                    PixelData::U8(vec![0; pixels * 2]),
                )
                .unwrap(),
                true,
            ),
            (
                Image::new(
                    width,
                    height,
                    PixelLayout::Rgb,
                    PixelData::U8(vec![0; pixels * 3]),
                )
                .unwrap(),
                false,
            ),
            (gray_extras, false),
        ];
        for (image, eligible) in cases {
            for enabled in [false, true] {
                for effort in [9, 10] {
                    let aliases = enabled && eligible;
                    let families = 5 - usize::from(aliases);
                    let full_encodes = families + if effort == 10 { 2 } else { 0 };
                    let mut options = options.clone();
                    options.scout_native_rct_aliases = enabled;
                    options.screen.limits.max_candidates = families;
                    options.full_limits.max_full_encodes = full_encodes;
                    options.full_limits.max_full_pixels = pixels as u64 * full_encodes as u64;
                    let mut probes = Vec::new();
                    let result = run_pixel_search(
                        &image,
                        &CandidateParams::default().backend(effort).unwrap(),
                        &options,
                        |params, _, _| {
                            Ok(complete(params, if family(params) >= 3 { 80 } else { 100 }))
                        },
                        |params, _| {
                            if params.predictor == Some(Predictor::Variable) {
                                probes.push(params.rct);
                            }
                            Ok(100)
                        },
                    )
                    .unwrap();
                    assert_eq!(result.report.full_encodes, full_encodes);
                    assert_eq!(
                        result.report.full_pixels,
                        pixels as u64 * full_encodes as u64
                    );
                    assert!(probes.contains(&Some(0)));
                    assert_eq!(probes.contains(&Some(6)), !aliases);
                    assert_eq!(
                        result.report.trials.iter().any(|t| t.params.rct == Some(6)),
                        !aliases
                    );
                    assert_eq!(
                        result.report.events.iter().any(|event| matches!(
                            event,
                            SearchEvent::Route {
                                name: "adaptive-scout-native-rct-aliases",
                                candidates: 1,
                            }
                        )),
                        aliases
                    );
                    assert_retained(&result, 3, 1, 80);
                }
            }
        }
    }

    #[test]
    fn native_rct_aliases_only_remove_exact_variable_rct_siblings() {
        let input = CandidateParams::default().backend(9).unwrap();
        let raw = lossless_probe_plan(&input, PlanOptions::default())
            .into_iter()
            .find(|c| c.params.predictor == Some(Predictor::Variable) && c.params.rct == Some(0))
            .unwrap();
        let mut cross = raw.params.clone();
        cross.rct = Some(6);
        let variants = [
            CandidateParams {
                learning_fraction: 0.2,
                ..cross.clone()
            },
            CandidateParams {
                group_size_shift: Some(2),
                ..cross.clone()
            },
            CandidateParams {
                max_properties: 0,
                ..cross.clone()
            },
            CandidateParams {
                palette_colors: 10000,
                ..cross.clone()
            },
            CandidateParams {
                predictor: Some(Predictor::Weighted),
                ..cross.clone()
            },
            CandidateParams {
                speed_tier: 0,
                ..cross.clone()
            },
            CandidateParams {
                rct: Some(17),
                ..cross.clone()
            },
            CandidateParams {
                rct: None,
                ..cross.clone()
            },
        ];
        // Keep the raw representative even when it follows its alias in the plan.
        let mut plan = vec![Candidate::new("alias", cross), raw];
        plan.extend(variants.into_iter().map(|p| Candidate::new("distinct", p)));
        plan.extend([
            Candidate::new("learned", literal(&input, 9, 0.01).unwrap()),
            Candidate::new("static", literal(&input, 9, 0.0).unwrap()),
        ]);
        let expected: Vec<_> = plan[1..].iter().map(|c| c.params.clone()).collect();
        assert_eq!(remove_native_rct_aliases(&mut plan, 1), 1);
        assert_eq!(
            plan.iter().map(|c| c.params.clone()).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(remove_native_rct_aliases(&mut plan, 1), 0);
    }

    #[test]
    fn native_rct_aliases_require_a_raw_representative() {
        let mut cross = CandidateParams::default().backend(9).unwrap();
        cross.predictor = Some(Predictor::Variable);
        cross.rct = Some(6);
        let mut plan = vec![Candidate::new("cross6-only", cross.clone())];
        assert_eq!(remove_native_rct_aliases(&mut plan, 2), 0);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].params, cross);
    }

    pub(super) fn family(params: &CandidateParams) -> usize {
        match (params.predictor, params.group_size_shift, params.rct) {
            (None, None, _) => 0,
            (None, Some(2), _) => 1,
            (Some(Predictor::Zero), _, _) => 2,
            (Some(Predictor::Variable), _, Some(0)) => 3,
            (Some(Predictor::Variable), _, Some(6)) => 4,
            _ => panic!("unexpected pilot candidate: {params:?}"),
        }
    }

    fn marker(family: usize, tier: i8) -> u8 {
        assert!(matches!(tier, 0 | 1));
        (family + 1) as u8 + if tier == 0 { 16 } else { 0 }
    }

    pub(super) fn complete(params: &CandidateParams, size: usize) -> TrialOutput<(usize, i8)> {
        let family = family(params);
        TrialOutput::Complete(EncodedOutput {
            bytes: vec![marker(family, params.speed_tier); size],
            artifact: (family, params.speed_tier),
        })
    }

    pub(super) fn assert_retained(
        result: &SearchResult<(usize, i8)>,
        id: usize,
        tier: i8,
        size: usize,
    ) {
        assert_eq!(result.bytes, vec![marker(id, tier); size]);
        assert_eq!(result.artifact, (id, tier));
        assert_eq!(family(&result.params), id);
        assert_eq!(result.params.speed_tier, tier);
        let winner = &result.report.trials[result.report.winner_index];
        assert_eq!(winner.params, result.params);
        assert_eq!(winner.bytes, Some(size));
        assert_eq!(winner.lower_bound, None);
    }

    #[test]
    fn flat_palette_keeps_static_tree_at_full_backend_effort() {
        let (image, mut options) = fixture();
        options.coverage_scan = true;
        let mut fractions = Vec::new();
        let result = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |params, _, budget| {
                assert_eq!(params.speed_tier, 1);
                if params.predictor != Some(Predictor::Zero) {
                    assert_eq!(budget.beat_bytes, Some(98));
                    return Ok(complete(params, 100));
                }
                if params.learning_fraction == 0.0 {
                    assert!(budget.beat_bytes.is_none());
                    Ok(complete(params, 99))
                } else {
                    assert_eq!(budget.beat_bytes, Some(99));
                    Ok(complete(params, 98))
                }
            },
            |params, _| {
                if params.predictor == Some(Predictor::Zero) {
                    fractions.push(params.learning_fraction);
                }
                Ok(
                    if params.predictor == Some(Predictor::Zero) && params.learning_fraction == 0.0
                    {
                        10
                    } else {
                        20
                    },
                )
            },
        )
        .unwrap();
        assert_eq!(result.bytes.len(), 98);
        assert!(fractions.contains(&0.0) && fractions.contains(&0.01));
    }

    #[test]
    fn finalist_wave_does_not_discard_candidates_before_learned_witness_commits() {
        for parallel in [false, true] {
            let (image, mut options) = fixture();
            options.coverage_scan = true;
            options.scout_parallel_finalists = parallel;
            let result = run_pixel_search(
                &image,
                &CandidateParams::default().backend(9).unwrap(),
                &options,
                |params, _, _| {
                    Ok(complete(
                        params,
                        match params.predictor {
                            Some(Predictor::Zero) if params.learning_fraction == 0.0 => 99,
                            Some(Predictor::Zero) => 98,
                            Some(Predictor::Variable) if params.rct == Some(6) => 80,
                            Some(Predictor::Variable) => 81,
                            _ => 100,
                        },
                    ))
                },
                |params, _| {
                    Ok(
                        if params.predictor == Some(Predictor::Zero)
                            && params.learning_fraction == 0.0
                        {
                            10
                        } else {
                            20
                        },
                    )
                },
            )
            .unwrap();
            assert_eq!(result.bytes.len(), 80);
            assert_eq!(result.params.rct, Some(6));
        }
    }

    #[test]
    fn first_output_is_real_and_all_probes_retain_effort() {
        let image = Image::new(
            1024,
            512,
            PixelLayout::Gray,
            PixelData::U8(vec![0; 1024 * 512]),
        )
        .unwrap();
        let mut options = DriverOptions::scout();
        options.scout_pilot_first = true;
        let result = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |params, _, budget| {
                assert_eq!(params.speed_tier, 1);
                assert!(budget.beat_bytes.is_none());
                assert_eq!(params.predictor, Some(Predictor::Zero));
                Ok(TrialOutput::Complete(EncodedOutput {
                    bytes: vec![19; 107],
                    artifact: 107,
                }))
            },
            |params, _| {
                assert_eq!(params.speed_tier, 1);
                Ok(
                    if params.predictor == Some(Predictor::Zero) && params.learning_fraction != 0.0
                    {
                        10
                    } else {
                        20
                    },
                )
            },
        )
        .unwrap();
        assert_eq!(result.bytes, vec![19; 107]);
        assert_eq!(result.report.full_encodes, 1);
        assert!(result.report.probe_encodes >= 5);
    }

    #[test]
    fn ambiguous_and_material_regional_winners_get_real_confirmation() {
        assert!(competitive(&[101, 102, 100], &[100, 100, 100]));
        assert!(competitive(&[80, 150, 150], &[100, 100, 100]));
        assert!(!competitive(&[115, 115, 115], &[100, 100, 100]));
        assert!(!competitive(&[1, 200], &[2, 100]));
        assert!(competitive(&[], &[]));
    }

    #[test]
    fn confidence_requires_three_consistent_regions_and_a_total_margin() {
        assert!(consistently_worse(&[1005, 1005, 1005], &[1000; 3]));
        assert!(!consistently_worse(&[1003, 1003, 1003], &[1000; 3]));
        assert!(!consistently_worse(&[1100, 1100, 999], &[1000; 3]));
        assert!(!consistently_worse(&[1005, 1005], &[1000; 2]));
        assert!(!consistently_worse(&[1, 1100, 1100], &[0, 1000, 1000]));
        assert!(!consistently_worse(&[], &[]));
    }

    #[test]
    fn e10_retains_e9_prefix_and_only_accepts_real_improvements() {
        let (image, options) = fixture();
        let sizes = [140, 130, 100, 120, 150];
        let mut e9_trace = Vec::new();
        let e9 = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |params, _, budget| {
                assert_eq!(params.speed_tier, 1);
                e9_trace.push((params.clone(), budget));
                Ok(complete(params, sizes[family(params)]))
            },
            |params, _| {
                assert_eq!(params.speed_tier, 1);
                Ok(100)
            },
        )
        .unwrap();
        assert_eq!(e9_trace.len(), 5);
        assert_retained(&e9, 2, 1, 100);

        for (deeper_literal, deeper_auto, winner, tier, size) in [
            (150, 160, 2, 1, 100),
            (100, 100, 2, 1, 100),
            (90, 110, 2, 0, 90),
            (95, 80, 0, 0, 80),
        ] {
            let mut trace = Vec::new();
            let e10 = run_pixel_search(
                &image,
                &CandidateParams::default().backend(10).unwrap(),
                &options,
                |params, _, budget| {
                    trace.push((params.clone(), budget));
                    let size = if params.speed_tier == 1 {
                        sizes[family(params)]
                    } else {
                        match family(params) {
                            2 => deeper_literal,
                            0 => deeper_auto,
                            id => panic!("unexpected deeper family: {id}"),
                        }
                    };
                    Ok(complete(params, size))
                },
                |params, _| {
                    assert_eq!(params.speed_tier, 1);
                    Ok(100)
                },
            )
            .unwrap();
            assert_eq!(&trace[..e9_trace.len()], e9_trace.as_slice());
            assert_eq!(trace.len(), 7);
            assert_eq!(trace[5].0, e9.params.backend(10).unwrap());
            assert_eq!(trace[6].0, CandidateParams::default().backend(10).unwrap());
            assert_eq!(trace[5].1.beat_bytes, Some(100));
            assert_eq!(trace[6].1.beat_bytes, Some(100.min(deeper_literal)));
            for (before, after) in e9.report.trials.iter().zip(&e10.report.trials) {
                assert_eq!(before.params, after.params);
                assert_eq!(before.stage, after.stage);
                assert_eq!(before.baseline, after.baseline);
                assert_eq!(before.bytes, after.bytes);
                assert_eq!(before.lower_bound, after.lower_bound);
            }
            assert!(e10.report.trials[5..].iter().all(|t| t.stage == 10));
            assert!(e10.bytes.len() <= e9.bytes.len());
            assert_retained(&e10, winner, tier, size);
        }
    }

    #[test]
    fn e10_confirms_rivals_screened_out_at_e9() {
        let (image, options) = fixture();
        let mut results = Vec::new();
        for effort in [9, 10] {
            results.push(
                run_pixel_search(
                    &image,
                    &CandidateParams::default().backend(effort).unwrap(),
                    &options,
                    |params, _, _| {
                        let size = if params.speed_tier == 0 {
                            200
                        } else {
                            [140, 130, 100, 120, 80][family(params)]
                        };
                        Ok(complete(params, size))
                    },
                    |params, _| {
                        assert_eq!(params.speed_tier, 1);
                        Ok(if family(params) == 4 { 200 } else { 100 })
                    },
                )
                .unwrap(),
            );
        }
        let e9 = &results[0];
        let e10 = &results[1];
        assert_eq!(e9.report.full_encodes, 4);
        assert_eq!(e10.report.full_encodes, 7);
        assert_retained(e9, 2, 1, 100);
        assert_retained(e10, 4, 1, 80);
        for (before, after) in e9.report.trials.iter().zip(&e10.report.trials) {
            assert_eq!(before.params, after.params);
            assert_eq!(before.bytes, after.bytes);
        }
        assert_eq!(e10.report.trials[4].stage, 9);
        assert_eq!(family(&e10.report.trials[4].params), 4);
    }

    #[test]
    fn bounded_challenger_keeps_completed_incumbent_and_its_probe_scores() {
        let (image, options) = fixture();
        let mut calls = Vec::new();
        let result = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |params, _, budget| {
                let id = family(params);
                calls.push(id);
                match id {
                    0 => {
                        assert_eq!(budget.beat_bytes, None);
                        Ok(complete(params, 100))
                    }
                    1 => {
                        assert_eq!(budget.beat_bytes, Some(100));
                        Ok(TrialOutput::Bounded { lower_bound: 137 })
                    }
                    _ => panic!("bounded candidate must not advance probe reference"),
                }
            },
            |params, _| Ok([100, 102, 104, 300, 300][family(params)]),
        )
        .unwrap();
        assert_eq!(calls, [0, 1]);
        assert_eq!(result.report.full_encodes, 2);
        assert_retained(&result, 0, 1, 100);
        let challenger = &result.report.trials[1];
        assert!(!challenger.baseline);
        assert_eq!(challenger.bytes, None);
        assert_eq!(challenger.lower_bound, Some(100));
    }

    #[test]
    fn pilot_rejects_incomplete_baseline_and_insufficient_challenger_bound() {
        let (image, options) = fixture();
        for incomplete_baseline in [false, true] {
            let mut calls = 0;
            let error = run_pixel_search(
                &image,
                &CandidateParams::default().backend(9).unwrap(),
                &options,
                |params, _, budget| {
                    calls += 1;
                    if family(params) == 0 {
                        assert_eq!(budget.beat_bytes, None);
                        if incomplete_baseline {
                            return Ok(TrialOutput::Bounded { lower_bound: 1000 });
                        }
                        return Ok(complete(params, 100));
                    }
                    assert_eq!(family(params), 1);
                    assert_eq!(budget.beat_bytes, Some(100));
                    Ok(TrialOutput::Bounded { lower_bound: 99 })
                },
                |params, _| Ok([100, 102, 300, 300, 300][family(params)]),
            )
            .unwrap_err();
            assert_eq!(calls, if incomplete_baseline { 1 } else { 2 });
            assert!(error.to_string().contains(if incomplete_baseline {
                "baseline/calibration callback returned an incomplete output"
            } else {
                "callback byte bound does not reach the incumbent"
            }));
        }
    }

    #[test]
    fn full_frame_rank_inversion_updates_reference_and_retains_exact_output() {
        let (image, options) = fixture();
        let mut calls = Vec::new();
        let result = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |params, _, budget| {
                let id = family(params);
                calls.push(id);
                assert!(id < 3);
                assert_eq!(budget.beat_bytes, [None, Some(500), Some(300)][id]);
                Ok(complete(params, [500, 300, 200][id]))
            },
            |params, _| Ok([100, 102, 104, 300, 300][family(params)]),
        )
        .unwrap();
        assert_eq!(calls, [0, 1, 2]);
        assert_eq!(result.report.full_encodes, 3);
        assert_eq!(result.report.winner_index, 2);
        assert_retained(&result, 2, 1, 200);
        assert!(result.report.events.iter().any(|event| matches!(
            event,
            SearchEvent::Route {
                name: "adaptive-scout-pilot-first",
                candidates: 3
            }
        )));
    }

    #[test]
    fn sample_and_full_ties_keep_plan_order_and_first_output() {
        let (image, options) = fixture();
        let mut calls = Vec::new();
        let result = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |params, _, budget| {
                assert_eq!(budget.beat_bytes, (!calls.is_empty()).then_some(250));
                calls.push(family(params));
                Ok(complete(params, 250))
            },
            |_, _| Ok(100),
        )
        .unwrap();
        assert_eq!(calls, [0, 1, 2, 3, 4]);
        assert_eq!(result.report.winner_index, 0);
        assert_eq!(result.report.full_encodes, 5);
        assert_retained(&result, 0, 1, 250);
        for (i, trial) in result.report.trials.iter().enumerate() {
            assert_eq!(trial.stage, 9);
            assert_eq!(trial.baseline, i == 0);
            assert_eq!(trial.bytes, Some(250));
            assert_eq!(trial.lower_bound, None);
        }
    }
}

#[cfg(test)]
#[path = "scout_pilot_staged_tests.rs"]
mod staged_tests;

#[cfg(test)]
#[path = "scout_initial_wave_tests.rs"]
mod initial_wave_tests;

#[cfg(test)]
#[path = "scout_overlap_tests.rs"]
mod overlap_tests;

#[cfg(test)]
#[path = "scout_near_tie_tests.rs"]
mod near_tie_tests;

#[cfg(test)]
#[path = "scout_overlap_finalists_tests.rs"]
mod overlap_finalists_tests;
