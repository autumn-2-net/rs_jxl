use super::super::{
    PaletteOptions, PaletteState, PaletteTools, apply_palette_range_with_options,
    attempt_with_admission,
};
use super::*;

fn channel(samples: Vec<i32>, width: usize) -> Channel {
    Channel {
        width,
        height: samples.len() / width,
        hshift: 0,
        vshift: 0,
        samples,
    }
}

fn target() -> Target {
    Target {
        stage: PaletteStage::PostRct,
        stream_id: 23,
        ordinal: 0,
    }
}

fn candidate(original: &[Channel]) -> (Vec<Channel>, Transform) {
    let mut candidate = original.to_vec();
    let transform = apply_palette_range_with_options(
        &mut candidate,
        &mut 0,
        16,
        0,
        original.len(),
        1024,
        true,
        PaletteMode::Absolute,
        PaletteTools::default(),
    )
    .unwrap()
    .unwrap();
    (candidate, transform)
}

#[test]
fn windows_are_bounded_contiguous_nonoverlapping_and_table_complete() {
    for population in [1, 255, 256, 257, 1024, 1025, 65536, 1 << 28] {
        let windows = windows(population, false);
        assert!(windows.len() <= WINDOWS);
        assert!(
            windows
                .iter()
                .all(|r| !r.is_empty() && r.len() <= WINDOW && r.end <= population)
        );
        assert!(windows.windows(2).all(|pair| pair[0].end <= pair[1].start));
        let sampled: usize = windows.iter().map(|r| r.len()).sum();
        assert_eq!(sampled, population.min(WINDOW * WINDOWS));
    }
    let table: Vec<_> = windows(TABLE_SAMPLES, true).into_iter().flatten().collect();
    assert_eq!(table, (0..TABLE_SAMPLES).collect::<Vec<_>>());
}

#[test]
fn native_pilot_counts_table_index_and_real_transform_header() {
    let original: Vec<_> = [11, 29, 37, 0]
        .into_iter()
        .map(|v| {
            channel(
                (0..2048)
                    .map(|i| if i % 2 == 0 { v } else { v + 1 })
                    .collect(),
                64,
            )
        })
        .collect();
    let saved = original.clone();
    let (candidate, transform) = candidate(&original);
    let (skip, after) = compare(&original, &candidate, &[], &[], &transform).unwrap();
    assert_eq!(original, saved);
    assert_eq!(skip.planes.len(), 4); // Includes alpha and invisible RGB.
    assert_eq!(after.planes.len(), 2);
    assert_eq!(after.planes[0].population, candidate[0].samples.len());
    assert_eq!(after.planes[0].sampled, after.planes[0].population);
    assert_eq!(after.planes[1].population, 2048);
    assert_eq!(after.planes[1].sampled, 1024);
    assert_eq!(
        after.transform_header_bits,
        header_bits(std::slice::from_ref(&transform)).unwrap()
    );
    for score in [&skip, &after] {
        assert!(score.total_bits.is_finite());
        assert_eq!(
            score.total_bits,
            score.planes.iter().map(|p| p.total_bits).sum::<f64>()
                + score.transform_header_bits as f64
        );
        assert!(
            score
                .planes
                .iter()
                .all(|p| p.leaves <= 4 && p.training_sites <= TRAIN_SITES)
        );
    }
    assert_eq!(
        (skip, after),
        compare(&original, &candidate, &[], &[], &transform).unwrap()
    );
}

#[test]
fn pilot_residuals_match_native_modular_narrowing_at_wide_edges() {
    let plane = channel(vec![i32::MIN, i32::MAX, -1, 0, 1, 1 << 30], 3);
    for position in 0..plane.samples.len() {
        let (n, properties) = site(&plane, position).unwrap();
        assert_eq!(properties[2], (position / plane.width) as i32);
        assert_eq!(properties[3], (position % plane.width) as i32);
        for predictor in PREDICTORS {
            assert_eq!(
                residual(plane.samples[position], n, predictor).unwrap(),
                (i64::from(plane.samples[position]) - predict_one(predictor, n, None).unwrap())
                    as i32
            );
        }
    }
    let score = plane_score(&plane, false).unwrap();
    assert!(score.total_bits.is_finite());
}

#[test]
fn veto_is_reject_only_shadow_and_forced_choices_win() {
    // The index is identical to the original; only table/header costs are added.
    let original = vec![channel(vec![0; 256], 32)];
    let (candidate, transform) = candidate(&original);
    let (skip, after) = compare(&original, &candidate, &[], &[], &transform).unwrap();
    assert!(
        after.total_bits > skip.total_bits,
        "unchanged zero index must pay the added table and metadata"
    );
    for mode in [Mode::Shadow, Mode::Veto] {
        for (allow_veto, forced, old) in [
            (true, false, true),
            (true, true, true),
            (false, false, true),
            (true, false, false),
        ] {
            let experiment = Experiment::new(mode, Some(target()));
            let mut cursor = Cursor::new(Some(&experiment), &[], target().stage, 23, allow_veto);
            let visit = cursor.next().unwrap();
            let vetoed = visit.compare(
                &original,
                &candidate,
                &[],
                &transform,
                PaletteMode::Absolute,
                old,
                forced,
            );
            assert_eq!(vetoed, mode == Mode::Veto && allow_veto && !forced && old);
            let record = experiment.events().remove(0);
            assert_eq!(record.old_would_accept, Some(old));
            assert_eq!(record.proxy_would_accept, Some(false));
            assert_eq!(record.forced, forced);
        }
    }
}

#[test]
fn hook_veto_rolls_back_and_default_has_no_experiment() {
    assert!(PaletteOptions::default().admission.is_none());
    // Zero preserves the original plane as the index, guaranteeing added cost.
    let original = vec![channel(vec![0; 256], 32)];
    for mode in [Mode::Shadow, Mode::Veto] {
        let experiment = Experiment::new(mode, Some(target()));
        let mut cursor = Cursor::new(Some(&experiment), &[], target().stage, 23, true);
        let mut channels = original.clone();
        let initial_state = PaletteState::new(8).unwrap();
        let mut state = initial_state;
        let mut transforms = Vec::new();
        let accepted = attempt_with_admission(
            &mut channels,
            &mut state,
            &mut transforms,
            8,
            0,
            1,
            2,
            true,
            PaletteMode::Absolute,
            PaletteTools::default(),
            0.0,
            true,
            None,
            None,
            cursor.next(),
        )
        .unwrap();
        assert_eq!(accepted, mode == Mode::Shadow);
        assert_eq!(experiment.events()[0].old_would_accept, Some(true));
        if !accepted {
            assert_eq!(channels, original);
            assert_eq!(state, initial_state);
            assert!(transforms.is_empty());
        }
    }
}

#[test]
fn table_budget_is_unavailable_not_a_veto_or_kernel_budget_change() {
    let original = vec![channel(
        (0..=TABLE_SAMPLES as i32).collect(),
        TABLE_SAMPLES + 1,
    )];
    let mut candidate = original.clone();
    let mut meta = 0;
    let transform = apply_palette_range_with_options(
        &mut candidate,
        &mut meta,
        16,
        0,
        1,
        (TABLE_SAMPLES + 1) as u32,
        true,
        PaletteMode::Absolute,
        PaletteTools::default(),
    )
    .unwrap()
    .unwrap();
    let experiment = Experiment::new(Mode::Veto, Some(target()));
    let mut cursor = Cursor::new(Some(&experiment), &[], target().stage, 23, true);
    assert!(!cursor.next().unwrap().compare(
        &original,
        &candidate,
        &[],
        &transform,
        PaletteMode::Absolute,
        true,
        false
    ));
    let record = experiment.events().remove(0);
    assert!(record.unavailable.unwrap().contains("budget"));
    assert!(record.proxy_would_accept.is_none());
    assert!(record.skip.is_none());
    assert!(record.candidate.is_none());
    assert_eq!(meta, 1);
}

#[test]
fn admission_target_identity_is_stream_local_and_counts_unselected_attempts() {
    let experiment = Experiment::new(
        Mode::Shadow,
        Some(Target {
            ordinal: 1,
            ..target()
        }),
    );
    let mut wrong = Cursor::new(Some(&experiment), &[], target().stage, 24, true);
    assert!(wrong.next().is_none());
    assert!(wrong.next().is_none());
    let mut selected = Cursor::new(Some(&experiment), &[], target().stage, 23, true);
    assert!(selected.next().is_none());
    let visit = selected.next().unwrap();
    assert_eq!(visit.target.ordinal, 1);
    assert!(selected.next().is_none());
}
