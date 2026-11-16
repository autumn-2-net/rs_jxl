//! Opt-in CfL peak selection, unchanged source fitting and integer reversibility.

use super::super::data::{Component, ComponentScanInfo, JpegData, QuantTable, ScanInfo};
use super::*;

fn differences(scores: &[i32; 256]) -> [i32; 257] {
    let mut result = [0; 257];
    result[0] = scores[0];
    for i in 1..256 {
        result[i] = scores[i] - scores[i - 1];
    }
    result[256] = -scores[255];
    result
}

fn expected_factor(scores: &[i32; 256]) -> i8 {
    let best = (0usize..255)
        .min_by_key(|&i| (-scores[i], i.abs_diff(127), i))
        .unwrap();
    if scores[best] <= scores[127] + 1 {
        0
    } else {
        i8::try_from(best as i32 - 127).unwrap()
    }
}

// Direct bin votes, independent of the fitter's difference-array accumulation.
fn tile_scores(layout: &TranscodeLayout<'_>, channel: usize, tx: u32, ty: u32) -> [i32; 256] {
    let mut scores = [0; 256];
    let (width, height) = layout.dimensions_in_blocks();
    let ratios = layout.planes()[channel].quantization().cfl_ratios();
    for y in ty * 8..height.min((ty + 1) * 8) {
        for x in tx * 8..width.min((tx + 1) * 8) {
            let BlockView::Jpeg(luma) = layout.planes()[1].block(x, y).unwrap() else {
                panic!("real luma");
            };
            let BlockView::Jpeg(side) = layout.planes()[channel].block(x, y).unwrap() else {
                panic!("real side plane");
            };
            for i in 1..64 {
                let m = f32::from(luma[i]) * (1.0f32 / 2048.0) * ratios[i] as f32;
                if m.abs() <= 1e-8f32 {
                    continue;
                }
                let s = 84.0f32 * f32::from(side[i]) + 127.0f32 * m;
                let radius = 84.0f32 * 0.5f32 * 0.9999f32;
                let a = (s - radius) / m;
                let b = (s + radius) / m;
                let start = a.min(b).ceil();
                let end = (a.max(b) + 1.0).floor();
                for (bin, score) in scores.iter_mut().enumerate() {
                    if bin as f32 >= start && (bin as f32) < end {
                        *score += 1;
                    }
                }
            }
        }
    }
    scores
}

fn fixture(width: u32, height: u32, sampling: &[(u32, u32)]) -> JpegData {
    let max_h = sampling.iter().map(|s| s.0).max().unwrap();
    let max_v = sampling.iter().map(|s| s.1).max().unwrap();
    JpegData {
        width,
        height,
        precision: 8,
        components: sampling
            .iter()
            .enumerate()
            .map(|(c, &(h, v))| {
                let width_in_blocks = width.div_ceil(8 * max_h) * h;
                let height_in_blocks = height.div_ceil(8 * max_v) * v;
                let length = width_in_blocks as usize * height_in_blocks as usize * BLOCK_SIZE;
                Component {
                    id: c as u32 + 1,
                    h_samp_factor: h,
                    v_samp_factor: v,
                    quant_index: c as u32,
                    width_in_blocks,
                    height_in_blocks,
                    coefficients: (0..length)
                        .map(|i| {
                            let y = [84, -84, 168, -168, 1, -1, 0][i % 7];
                            match c {
                                0 => y,
                                1 => y / 2,
                                _ => -y / 4,
                            }
                        })
                        .collect(),
                }
            })
            .collect(),
        quant: (0..sampling.len())
            .map(|c| QuantTable {
                index: c as u32,
                values: [1; BLOCK_SIZE],
                ..Default::default()
            })
            .collect(),
        scan_info: vec![ScanInfo {
            components: (0..sampling.len())
                .map(|component_index| ComponentScanInfo {
                    component_index,
                    ..Default::default()
                })
                .collect(),
            se: 63,
            quant_table_indices: [Some(0), Some(1), Some(2), None],
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[test]
fn separated_peaks_select_a_real_maximum_and_deterministic_nearest_zero() {
    let mut scores = [0; 256];
    scores[10] = 6;
    scores[20] = 6;
    let votes = differences(&scores);
    assert_eq!(choose_factor(&votes), -112);
    assert_eq!(choose_actual_maximum_factor(&votes), -107);

    scores.fill(0);
    scores[100] = 6;
    scores[154] = 6;
    assert_eq!(choose_actual_maximum_factor(&differences(&scores)), -27);
    scores[153] = 6;
    assert_eq!(choose_actual_maximum_factor(&differences(&scores)), 26);

    scores.fill(0);
    scores[160..=170].fill(6);
    assert_eq!(choose_actual_maximum_factor(&differences(&scores)), 33);
}

#[test]
fn endpoint_255_is_never_scored_as_wrapped_minus_128() {
    let mut scores = [0; 256];
    scores[255] = 9;
    assert_eq!(choose_factor(&differences(&scores)), -128);
    assert_eq!(choose_actual_maximum_factor(&differences(&scores)), 0);
    scores[254] = 4;
    assert_eq!(choose_actual_maximum_factor(&differences(&scores)), 127);
    scores[0] = 5;
    assert_eq!(choose_actual_maximum_factor(&differences(&scores)), -127);

    for bin in 0..255 {
        scores.fill(0);
        scores[bin] = 3;
        let factor = choose_actual_maximum_factor(&differences(&scores));
        assert_eq!(i32::from(factor), bin as i32 - 127);
        assert_eq!((i32::from(factor) + 127) as usize, bin);
    }
}

#[test]
fn empty_votes_and_strict_zero_improvement_gate_are_preserved() {
    let mut scores = [0; 256];
    assert_eq!(choose_actual_maximum_factor(&differences(&scores)), 0);
    scores[127] = 5;
    scores[169] = 6;
    assert_eq!(choose_actual_maximum_factor(&differences(&scores)), 0);
    scores[169] = 7;
    assert_eq!(choose_actual_maximum_factor(&differences(&scores)), 42);
    scores[127] = 7;
    assert_eq!(choose_actual_maximum_factor(&differences(&scores)), 0);
}

#[test]
fn peak_scan_matches_independent_score_ordering() {
    let mut state = 29u32;
    for _ in 0..256 {
        let scores = std::array::from_fn(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 16) % 31) as i32
        });
        assert_eq!(
            choose_actual_maximum_factor(&differences(&scores)),
            expected_factor(&scores)
        );
    }
}

#[test]
fn default_policy_and_compute_preserve_source_maps_and_residuals() {
    assert_eq!(CflFitPolicy::default(), CflFitPolicy::Source);
    let mut jpeg = fixture(73, 65, &[(1, 1); 3]);
    jpeg.components[0].coefficients[1..64].fill(84);
    jpeg.components[1].coefficients[1..64].fill(128);
    let layout = TranscodeLayout::new(&jpeg).unwrap();
    for mode in [CflMode::Disabled, CflMode::Enabled] {
        let baseline = execution::serial(|| JpegCfl::compute(&layout, mode));
        let explicit =
            execution::serial(|| JpegCfl::compute_with_policy(&layout, mode, CflFitPolicy::Source));
        assert_eq!(baseline.is_enabled(), explicit.is_enabled());
        assert_eq!(
            baseline.dimensions_in_tiles(),
            explicit.dimensions_in_tiles()
        );
        assert_eq!(baseline.ytox_map(), explicit.ytox_map());
        assert_eq!(baseline.ytob_map(), explicit.ytob_map());
        for channel in 0..3 {
            let (width, height) = layout.planes()[channel].dimensions_in_blocks();
            for y in 0..height {
                for x in 0..width {
                    let mut expected = [0; 64];
                    let mut actual = [0; 64];
                    baseline
                        .forward_block(channel, x, y, &mut expected)
                        .unwrap();
                    explicit.forward_block(channel, x, y, &mut actual).unwrap();
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[test]
fn fitted_endpoint_changes_only_the_opt_in_policy() {
    let mut jpeg = fixture(8, 8, &[(1, 1); 3]);
    jpeg.components[0].coefficients[1..].fill(84);
    jpeg.components[1].coefficients[1..].fill(128);
    let layout = TranscodeLayout::new(&jpeg).unwrap();
    let source = execution::serial(|| JpegCfl::compute(&layout, CflMode::Enabled));
    let actual = execution::serial(|| {
        JpegCfl::compute_with_policy(&layout, CflMode::Enabled, CflFitPolicy::ActualMaximum)
    });
    assert_eq!(source.ytox_map(), &[-128]);
    assert_eq!(actual.ytox_map(), &[0]);
    let mut residual = [0; 64];
    actual.forward_block(0, 0, 0, &mut residual).unwrap();
    assert_eq!(&residual[1..], &[128; 63]);
    let mut restored = [0; 64];
    actual
        .inverse_block(0, 0, 0, &residual, &mut restored)
        .unwrap();
    assert_eq!(restored.as_slice(), jpeg.components[1].coefficients);
}

#[test]
fn selected_factors_retain_signed_division_and_asymmetric_rounding() {
    for (bin, expected) in [(169, [1, 0, 2, -1]), (85, [0, 1, -1, 2])] {
        let mut scores = [0; 256];
        scores[bin] = 3;
        let factor = choose_actual_maximum_factor(&differences(&scores));
        for (luma, expected) in [1, -1, 3, -3].into_iter().zip(expected) {
            assert_eq!(prediction(luma, 2048, factor, 1).unwrap(), expected);
        }
    }
    assert_eq!(ratio_jpeg(-1), -24);
    assert_eq!(ratio_jpeg(1), 24);
    for factor in [-127, -42, -1, 0, 1, 42, 127] {
        for ratio in [0, 1, 1537, 2731, CFL_FIXED_POINT_RATIO_MAX] {
            let scale = (i64::from(factor) * 2048 / 84 * i64::from(ratio) + 1024).div_euclid(2048);
            for luma in [-1024i16, -3, -1, 0, 1, 3, 1024] {
                let expected = (i64::from(luma) * scale + 1024).div_euclid(2048);
                assert_eq!(
                    i64::from(prediction(luma, ratio, factor, 1).unwrap()),
                    expected
                );
            }
        }
    }
}

#[test]
fn actual_maps_round_trip_all_coefficients_across_partial_tiles() {
    let mut jpeg = fixture(73, 65, &[(1, 1); 3]);
    for (i, table) in jpeg.quant.iter_mut().enumerate() {
        table.values[1..].fill(i as i32 + 1);
    }
    let layout = TranscodeLayout::new(&jpeg).unwrap();
    let cfl = execution::serial(|| {
        JpegCfl::compute_with_policy(&layout, CflMode::Enabled, CflFitPolicy::ActualMaximum)
    });
    assert_eq!(cfl.dimensions_in_tiles(), (2, 2));
    assert!(cfl.ytox_map().iter().any(|&factor| factor != 0));
    for channel in [0, 2] {
        for ty in 0..2 {
            for tx in 0..2 {
                assert_eq!(
                    cfl.factor(channel, tx, ty),
                    Some(expected_factor(&tile_scores(&layout, channel, tx, ty)))
                );
            }
        }
    }
    for channel in 0..3 {
        let plane = &layout.planes()[channel];
        let (width, height) = plane.dimensions_in_blocks();
        for y in 0..height {
            for x in 0..width {
                let BlockView::Jpeg(original) = plane.block(x, y).unwrap() else {
                    panic!("three real planes");
                };
                let BlockView::Jpeg(luma) = layout.planes()[1].block(x, y).unwrap() else {
                    panic!("real luma");
                };
                let factor = cfl.factor(channel, x / 8, y / 8).unwrap();
                let mut residual = [0; 64];
                cfl.forward_block(channel, x, y, &mut residual).unwrap();
                assert_eq!(residual[0], i32::from(original[0]));
                for i in 1..64 {
                    let ratio = i64::from(plane.quantization().cfl_ratios()[i]);
                    let scale = (i64::from(factor) * 2048 / 84 * ratio + 1024).div_euclid(2048);
                    let predicted = (i64::from(luma[i]) * scale + 1024).div_euclid(2048);
                    assert_eq!(
                        i64::from(residual[transpose(i)]),
                        i64::from(original[i]) - predicted
                    );
                }
                let mut restored = [0; 64];
                cfl.inverse_block(channel, x, y, &residual, &mut restored)
                    .unwrap();
                assert_eq!(&restored, original);
            }
        }
    }
}

#[test]
fn actual_policy_retains_disabled_gray_and_subsampling_guards() {
    for sampling in [
        &[(1, 1); 3][..],
        &[(2, 2), (1, 1), (1, 1)],
        &[(2, 1), (1, 1), (1, 1)],
        &[(1, 2), (1, 1), (1, 1)],
        &[(1, 1)],
    ] {
        let jpeg = fixture(17, 9, sampling);
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        for mode in [CflMode::Disabled, CflMode::Enabled] {
            let cfl = execution::serial(|| {
                JpegCfl::compute_with_policy(&layout, mode, CflFitPolicy::ActualMaximum)
            });
            let enabled = sampling == [(1, 1); 3] && mode == CflMode::Enabled;
            assert_eq!(cfl.is_enabled(), enabled);
            if enabled {
                continue;
            }
            assert!(cfl.ytox_map().iter().chain(cfl.ytob_map()).all(|&f| f == 0));
            for channel in 0..3 {
                let plane = &layout.planes()[channel];
                let (width, height) = plane.dimensions_in_blocks();
                let mut expected = [0; 64];
                plane
                    .block(width - 1, height - 1)
                    .unwrap()
                    .write_transposed(&mut expected);
                let mut actual = [0; 64];
                cfl.forward_block(channel, width - 1, height - 1, &mut actual)
                    .unwrap();
                assert_eq!(actual, expected);
                let mut restored = [0; 64];
                cfl.inverse_block(channel, width - 1, height - 1, &actual, &mut restored)
                    .unwrap();
                for i in 0..64 {
                    assert_eq!(i32::from(restored[i]), expected[transpose(i)]);
                }
            }
        }
    }
}

#[test]
fn actual_policy_keeps_rgb_eligible_with_its_original_channel_order() {
    let mut jpeg = fixture(8, 8, &[(1, 1); 3]);
    for (component, &id) in jpeg.components.iter_mut().zip(b"RGB") {
        component.id = u32::from(id);
    }
    jpeg.components[1].coefficients[1..].fill(84);
    jpeg.components[0].coefficients[1..].fill(-42);
    jpeg.components[2].coefficients[1..].fill(21);
    let layout = TranscodeLayout::new(&jpeg).unwrap();
    assert_eq!(layout.jpeg_order(), [0, 1, 2]);
    let cfl = execution::serial(|| {
        JpegCfl::compute_with_policy(&layout, CflMode::Enabled, CflFitPolicy::ActualMaximum)
    });
    assert!(cfl.is_enabled());
    assert_eq!(cfl.ytox_map(), &[-42]);
    assert_eq!(cfl.ytob_map(), &[21]);
    for channel in 0..3 {
        let mut residual = [0; 64];
        cfl.forward_block(channel, 0, 0, &mut residual).unwrap();
        let mut restored = [0; 64];
        cfl.inverse_block(channel, 0, 0, &residual, &mut restored)
            .unwrap();
        assert_eq!(restored.as_slice(), jpeg.components[channel].coefficients);
    }
}

#[test]
fn actual_policy_matches_serial_maps_in_caller_owned_pool() {
    let jpeg = fixture(73, 129, &[(1, 1); 3]);
    let layout = TranscodeLayout::new(&jpeg).unwrap();
    let fit =
        || JpegCfl::compute_with_policy(&layout, CflMode::Enabled, CflFitPolicy::ActualMaximum);
    let expected = execution::serial(fit);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let actual = execution::with_pool(&pool, fit);
    assert_eq!(actual.ytox_map(), expected.ytox_map());
    assert_eq!(actual.ytob_map(), expected.ytob_map());
}
