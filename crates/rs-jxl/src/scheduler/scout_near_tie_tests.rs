use super::tests::{complete, family, fixture};
use super::*;

const AUTO: [usize; 3] = [26060, 35349, 47677];
const YC: [usize; 3] = [27898, 36416, 47270];

#[test]
fn near_tie_integer_boundaries_require_complete_nonzero_scores() {
    let leader = [10000; 3];
    for (candidate, expected) in [
        ([9900, 10125, 10125], true),
        ([9899, 10126, 10125], false),
        ([9900, 10125, 10124], false),
        ([10000, 10075, 10075], true),
        ([10000, 10074, 10075], false),
        ([9800, 10175, 10175], false),
        ([10000; 3], false),
        ([0, 20000, 20000], false),
    ] {
        assert_eq!(
            near_tie_worse(&candidate, &leader),
            expected,
            "{candidate:?}"
        );
    }
    assert!(!near_tie_worse(&[], &[]));
    assert!(!near_tie_worse(&[10100; 2], &[10000; 2]));
    assert!(!near_tie_worse(&[10100; 3], &[10000; 4]));
    assert!(!near_tie_worse(&[10100; 3], &[0, 10000, 10000]));
    assert!(near_tie_worse(&[9900, 10100, 10100, 10100], &[10000; 4]));
}

#[test]
fn near_tie_widening_and_aggregate_comparison_do_not_overflow() {
    let unit = usize::MAX / 201;
    assert!(near_tie_worse(&[unit * 201; 3], &[unit * 200; 3]));
    assert!(!near_tie_worse(&[usize::MAX; 3], &[usize::MAX; 3]));
    let unit = u128::MAX / 201;
    assert!(half_percent_worse(unit * 201, unit * 200));
    assert!(!half_percent_worse(unit * 201 - 1, unit * 200));
    assert!(!half_percent_worse(u128::MAX, u128::MAX));
    assert!(half_percent_worse(u128::MAX, u128::MAX / 2));
    assert!(!half_percent_worse(199, 199));
    assert!(half_percent_worse(200, 199));
    assert!(half_percent_worse(201, 200));
    assert!(!half_percent_worse(201, 201));
    assert!(!half_percent_worse(202, 201));
    assert!(half_percent_worse(203, 201));
}

#[test]
fn near_tie_is_independent_opt_in_and_never_mutates_scores() {
    let automatic = CandidateParams::default().backend(9).unwrap();
    let mut challenger = automatic.clone();
    challenger.predictor = Some(Predictor::Variable);
    challenger.rct = Some(6);
    let leader = AUTO.to_vec();
    let scores = YC.to_vec();
    assert!(!consistently_worse(&scores, &leader));
    assert!(competitive(&scores, &leader));
    assert!(near_tie_worse(&scores, &leader));
    for old in [false, true] {
        assert!(pilot_eligible(
            9,
            old,
            false,
            &automatic,
            &leader,
            &challenger,
            &scores
        ));
        assert!(!pilot_eligible(
            9,
            old,
            true,
            &automatic,
            &leader,
            &challenger,
            &scores
        ));
        assert!(pilot_eligible(
            10,
            old,
            true,
            &automatic,
            &leader,
            &challenger,
            &scores
        ));
    }
    assert_eq!(leader, AUTO);
    assert_eq!(scores, YC);

    let leader = [22395, 24096, 21172];
    let scores = [22526, 24531, 21632];
    assert!(consistently_worse(&scores, &leader));
    assert!(pilot_eligible(
        9,
        false,
        false,
        &automatic,
        &leader,
        &challenger,
        &scores
    ));
    assert!(!pilot_eligible(
        9,
        true,
        false,
        &automatic,
        &leader,
        &challenger,
        &scores
    ));
    assert!(!pilot_eligible(
        9,
        false,
        true,
        &automatic,
        &leader,
        &challenger,
        &scores
    ));
}

#[test]
fn near_tie_preserves_literal_witnesses_explicit_leaders_and_regional_wins() {
    let automatic = CandidateParams::default().backend(9).unwrap();
    for fraction in [0.0, 0.01] {
        let literal = literal(&automatic, 9, fraction).unwrap();
        assert!(pilot_eligible(
            9, true, true, &automatic, &AUTO, &literal, &YC
        ));
    }
    let fixed = literal(&automatic, 9, 0.0).unwrap();
    let learned = literal(&automatic, 9, 0.01).unwrap();
    assert!(pilot_eligible(
        9, true, true, &fixed, &[10; 3], &learned, &[1000; 3]
    ));

    let mut variable = automatic.clone();
    variable.predictor = Some(Predictor::Variable);
    for predictor in [Predictor::Best, Predictor::Variable, Predictor::Zero] {
        let mut leader = automatic.clone();
        leader.predictor = Some(predictor);
        assert!(pilot_eligible(
            9, true, true, &leader, &AUTO, &variable, &YC
        ));
    }
    assert!(pilot_eligible(
        9,
        true,
        true,
        &automatic,
        &[10000; 3],
        &variable,
        &[9800, 10175, 10175]
    ));
    assert!(pilot_eligible(
        10,
        true,
        true,
        &automatic,
        &[10000; 3],
        &variable,
        &[20000; 3]
    ));
}

#[test]
fn near_tie_default_and_pilot_dependency_are_explicit() {
    assert!(!DriverOptions::default().scout_near_tie_consensus);
    assert!(!DriverOptions::scout().scout_near_tie_consensus);
    let (image, mut options) = fixture();
    options.scout_pilot_first = false;
    options.scout_near_tie_consensus = true;
    let error = run_pixel_search::<_, _, ()>(
        &image,
        &CandidateParams::default().backend(9).unwrap(),
        &options,
        |_, _, _| panic!("invalid policy must not encode"),
        |_, _| panic!("invalid policy must not probe"),
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput(message)
        if message == "pilot refinements require pilot-first"));
}

#[test]
fn near_tie_skips_only_the_challenger_full_and_keeps_real_bounds() {
    let (image, mut options) = fixture();
    options.screen.regions.full_width_strips = true;
    options.screen.regions.strip_rows = Some(32);
    for old in [false, true] {
        for enabled in [false, true] {
            options.scout_pilot_consensus = old;
            options.scout_near_tie_consensus = enabled;
            let mut calls = Vec::new();
            let mut regions = [0usize; 5];
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
                            Ok(complete(params, 717614))
                        }
                        4 => {
                            assert!(!enabled);
                            assert_eq!(budget.beat_bytes, Some(717614));
                            Ok(TrialOutput::Bounded {
                                lower_bound: 717614,
                            })
                        }
                        _ => panic!("noncompetitive family"),
                    }
                },
                |params, _| {
                    let id = family(params);
                    let region = regions[id];
                    regions[id] += 1;
                    Ok(match id {
                        0 => AUTO[region],
                        4 => YC[region],
                        _ => 100000,
                    })
                },
            )
            .unwrap();
            assert_eq!(calls, if enabled { vec![0] } else { vec![0, 4] });
            assert_eq!(regions, [3; 5]);
            assert_eq!(result.bytes.len(), 717614);
            assert_eq!(result.report.full_encodes, if enabled { 1 } else { 2 });
            assert_eq!(result.report.probe_encodes, 15);
            let frontier = result
                .report
                .events
                .iter()
                .find_map(|e| match e {
                    SearchEvent::Frontier { candidates } => Some(candidates),
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                frontier
                    .iter()
                    .find(|c| family(&c.params) == 0)
                    .unwrap()
                    .context_bytes,
                Some(AUTO.iter().sum())
            );
            assert_eq!(
                frontier
                    .iter()
                    .find(|c| family(&c.params) == 4)
                    .unwrap()
                    .context_bytes,
                Some(YC.iter().sum())
            );
        }
    }
}

#[test]
fn low_color_scan_keeps_a_smaller_full_candidate_despite_near_tie_probes() {
    let (image, mut options) = fixture();
    options.coverage_scan = true;
    options.scout_pilot_consensus = false;
    options.scout_near_tie_consensus = true;
    options.screen.regions.full_width_strips = true;
    options.screen.regions.strip_rows = Some(32);
    let mut regions = [0usize; 5];
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
                    Ok(complete(params, 83748))
                }
                1 => {
                    assert_eq!(budget.beat_bytes, Some(83748));
                    Ok(complete(params, 83134))
                }
                _ => panic!("noncompetitive family"),
            }
        },
        |params, _| {
            let id = family(params);
            let region = regions[id];
            regions[id] += 1;
            Ok(match id {
                0 => [13153, 8474, 2390][region],
                1 => [13274, 8680, 2455][region],
                _ => 100000,
            })
        },
    )
    .unwrap();
    assert_eq!(calls, [0, 1]);
    assert_eq!(result.bytes.len(), 83134);
    assert_eq!(result.report.full_encodes, 2);
}
