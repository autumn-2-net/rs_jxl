use super::super::predictor::NeighborRow;
use super::*;

fn headers() -> Vec<Header> {
    let mut result: Vec<_> = (0..5).map(|mode| Header::preset(mode).unwrap()).collect();
    result.extend([
        Header {
            p1_c: 31,
            p2_c: 31,
            p3_ca: 31,
            p3_cb: 31,
            p3_cc: 31,
            p3_cd: 31,
            p3_ce: 31,
            w: [15; 4],
        },
        Header {
            w: [0; 4],
            ..Header::default()
        },
        Header {
            w: [0, 15, 0, 1],
            ..Header::default()
        },
    ]);
    result
}

fn neighbors(samples: &[i32], width: usize, x: usize, y: usize) -> [i32; 5] {
    let n = NeighborRow::new(samples, width, width, y)
        .unwrap()
        .at(x)
        .unwrap();
    [n.top, n.left, n.topright, n.topleft, n.toptop]
}

fn compare_step(
    actual: &mut State,
    reference: &mut State,
    value: i32,
    (x, y): (usize, usize),
    [n, w, ne, nw, nn]: [i32; 5],
) -> (i32, i32) {
    let expected = reference.predict(x, y, n, w, ne, nw, nn).unwrap();
    assert!(reference.pending);
    let output = actual
        .predict_and_update(value, x, y, n, w, ne, nw, nn)
        .unwrap();
    reference.update(value, x, y).unwrap();
    assert_eq!(output, expected, "prediction/property at ({x}, {y})");
    assert!(!actual.pending);
    assert_eq!(
        actual, reference,
        "complete post-update state at ({x}, {y})"
    );
    output
}

fn rejects_without_mutation(state: &mut State, x: usize, y: usize) {
    let before = state.clone();
    assert!(matches!(
        state.predict_and_update(i32::MIN, x, y, 0, 0, 0, 0, 0),
        Err(Error::InvalidInput(_))
    ));
    assert_eq!(*state, before);
}

#[test]
fn fused_rasters_match_pair_predictions_properties_and_entire_state() {
    for header in headers() {
        for width in [1, 2, 3, 7, 16, 33] {
            for pattern in 0..8 {
                let mut random = 0x8e41_5a17u32;
                let samples: Vec<_> = (0..width * 24)
                    .map(|p| {
                        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                        match pattern {
                            0 => 0,
                            1 => -17,
                            2 => i32::MIN,
                            3 => i32::MAX,
                            4 => {
                                if p / width == 6 && p % width == width / 2 {
                                    123
                                } else {
                                    0
                                }
                            }
                            5 => {
                                if (p / width + p % width) % 2 == 0 {
                                    i32::MIN
                                } else {
                                    i32::MAX
                                }
                            }
                            6 => p as i32 - 127,
                            _ => random as i32,
                        }
                    })
                    .collect();
                let mut actual = State::new(header, width).unwrap();
                let mut reference = actual.clone();
                let error_ptr = actual.error.as_ptr();
                let pred_ptrs = actual.pred_errors.each_ref().map(|v| v.as_ptr());
                for (p, &value) in samples.iter().enumerate() {
                    let (x, y) = (p % width, p / width);
                    compare_step(
                        &mut actual,
                        &mut reference,
                        value,
                        (x, y),
                        neighbors(&samples, width, x, y),
                    );
                }
                assert_eq!(actual.error.as_ptr(), error_ptr);
                assert_eq!(actual.pred_errors.each_ref().map(|v| v.as_ptr()), pred_ptrs);
                actual.reset();
                reference.reset();
                assert_eq!(actual, reference);
                assert_eq!(actual, State::new(header, width).unwrap());
            }
        }
    }
}

#[test]
fn fused_wrapped_histories_and_flat_actual_mismatches_keep_every_field() {
    for header in headers() {
        for width in [1, 2, 17] {
            for y in [0, 1, 2, usize::MAX] {
                for x in 0..width {
                    let mut initial = State::new(header, width).unwrap();
                    initial.next = Some((x, y));
                    for (i, errors) in initial.pred_errors.iter_mut().enumerate() {
                        for (p, error) in errors.iter_mut().enumerate() {
                            *error = [0, u32::MAX, u32::MAX - 1, 1, 1 << 31][(i + p) % 5];
                        }
                    }
                    for signed_history in [false, true] {
                        for (i, error) in initial.error.iter_mut().enumerate() {
                            *error = if signed_history {
                                [i32::MIN, -9, 9, i32::MAX][i % 4]
                            } else {
                                0
                            };
                        }
                        for n in [0, i32::MIN, i32::MAX] {
                            for ns in [[n; 5], [n, !n, n.wrapping_add(1), 9, -9]] {
                                let mut actual = initial.clone();
                                let mut reference = initial.clone();
                                compare_step(&mut actual, &mut reference, !n, (x, y), ns);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fused_returns_old_error_property_before_updating_histories() {
    let mut actual = State::new(Header::default(), 3).unwrap();
    actual.next = Some((1, 1));
    let (cur, prev) = actual.rows(1);
    actual.error[cur] = 8;
    actual.error[prev + 1] = -9;
    actual.error[prev] = 9;
    actual.error[prev + 2] = 8;
    let mut reference = actual.clone();
    let output = compare_step(&mut actual, &mut reference, 10000, (1, 1), [0; 5]);
    assert_eq!(output.1, -9, "N wins the equal-magnitude N/NW tie");
    assert_ne!(actual.error[cur + 1], output.1);
    assert_ne!(actual.pred_errors[0][prev + 2], 0);
}

#[test]
fn fused_rejects_bad_coordinates_and_pending_predictions_without_mutation() {
    for width in [1, 2, 17] {
        let mut state = State::new(Header::default(), width).unwrap();
        for (x, y) in [(1, 0), (width, 0), (0, 1), (usize::MAX, usize::MAX)] {
            rejects_without_mutation(&mut state, x, y);
        }
        state.predict(0, 0, 0, 0, 0, 0, 0).unwrap();
        for (x, y) in [(0, 0), (1, 0), (0, 1), (usize::MAX, usize::MAX)] {
            rejects_without_mutation(&mut state, x, y);
        }
        state.update(7, 0, 0).unwrap();
        rejects_without_mutation(&mut state, 0, 0);
        let (x, y) = state.next.unwrap();
        let mut reference = state.clone();
        compare_step(&mut state, &mut reference, -13, (x, y), [7; 5]);
    }
}

#[test]
fn fused_and_public_pairs_can_alternate_and_reset_pending_state() {
    for header in headers() {
        let width = 3;
        let samples: Vec<_> = (0..width * 9).map(|p| p as i32 * 47 - 333).collect();
        let mut actual = State::new(header, width).unwrap();
        let mut reference = actual.clone();
        for (p, &value) in samples.iter().enumerate() {
            let (x, y) = (p % width, p / width);
            let ns @ [n, w, ne, nw, nn] = neighbors(&samples, width, x, y);
            if p % 2 == 0 {
                compare_step(&mut actual, &mut reference, value, (x, y), ns);
                let before = actual.clone();
                assert!(actual.update(value, x, y).is_err());
                assert_eq!(actual, before);
            } else {
                assert_eq!(
                    actual.predict(x, y, n, w, ne, nw, nn).unwrap(),
                    reference.predict(x, y, n, w, ne, nw, nn).unwrap()
                );
                assert_eq!(actual, reference);
                rejects_without_mutation(&mut actual, x, y);
                actual.update(value, x, y).unwrap();
                reference.update(value, x, y).unwrap();
                assert_eq!(actual, reference);
            }
        }
        actual.predict(0, 9, 7, 7, 7, 7, 7).unwrap();
        actual.reset();
        reference.reset();
        assert_eq!(actual, reference);
        compare_step(&mut actual, &mut reference, 42, (0, 0), [0; 5]);
    }
}

#[test]
fn fused_coordinate_exhaustion_matches_pair_and_never_wraps() {
    for width in [1, 2, 3] {
        let mut actual = State::new(Header::preset(4).unwrap(), width).unwrap();
        actual.next = Some((0, usize::MAX - 1));
        let mut reference = actual.clone();
        for y in [usize::MAX - 1, usize::MAX] {
            for x in 0..width {
                compare_step(
                    &mut actual,
                    &mut reference,
                    x as i32 - 19,
                    (x, y),
                    [7, -3, 13, -19, 23],
                );
            }
        }
        assert_eq!(actual.next, None);
        rejects_without_mutation(&mut actual, 0, 0);
        rejects_without_mutation(&mut actual, 0, usize::MAX);
        actual.reset();
        reference.reset();
        compare_step(&mut actual, &mut reference, i32::MIN, (0, 0), [0; 5]);
    }
}

#[test]
fn fused_shifted_stripe_warmup_and_unsampled_pixels_keep_state() {
    let width = 9;
    let height = 50;
    let samples: Vec<_> = (0..width * height)
        .map(|p| {
            if p / width % 7 == 0 {
                0
            } else {
                (p as i32).wrapping_mul(1_000_003)
            }
        })
        .collect();
    for header in headers() {
        for begin in [0usize, 1, 16, 17, 33] {
            let warm_begin = begin.saturating_sub(16);
            let end = (begin + 11).min(height);
            let mut actual = State::new(header, width).unwrap();
            let mut reference = actual.clone();
            let mut selected = 0;
            for y in warm_begin..end {
                for x in 0..width {
                    let pos = y * width + x;
                    compare_step(
                        &mut actual,
                        &mut reference,
                        samples[pos],
                        (x, y - warm_begin),
                        neighbors(&samples, width, x, y),
                    );
                    if y >= begin && pos % 5 == 0 {
                        selected += 1;
                    }
                }
            }
            assert!(selected > 0);
            assert_eq!(actual.next, Some((0, end - warm_begin)));
        }
    }
}
