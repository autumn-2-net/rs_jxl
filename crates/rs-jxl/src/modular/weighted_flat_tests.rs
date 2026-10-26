use super::*;

fn headers() -> Vec<Header> {
    let mut headers: Vec<_> = (0..5).map(|mode| Header::preset(mode).unwrap()).collect();
    headers.extend([
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
    headers
}

fn compare_prediction(
    actual: &mut State,
    reference: &mut State,
    x: usize,
    y: usize,
    [n, w, ne, nw, nn]: [i32; 5],
) -> (i32, i32) {
    let expected = reference
        .predict_impl::<false>(x, y, n, w, ne, nw, nn)
        .unwrap();
    let result = actual.predict(x, y, n, w, ne, nw, nn).unwrap();
    assert_eq!(result, expected, "prediction at ({x}, {y})");
    assert_eq!(actual, reference, "pending state at ({x}, {y})");
    result
}

fn compare_update(actual: &mut State, reference: &mut State, value: i32, x: usize, y: usize) {
    actual.update(value, x, y).unwrap();
    reference.update(value, x, y).unwrap();
    assert_eq!(actual, reference, "updated state at ({x}, {y})");
}

fn signed_positions(state: &State, x: usize, y: usize) -> [Option<usize>; 4] {
    let (cur, prev) = state.rows(y);
    [
        x.checked_sub(1).map(|west| cur + west),
        Some(prev + x),
        Some(prev + x.saturating_sub(1)),
        Some(prev + (x + 1).min(state.width - 1)),
    ]
}

#[test]
fn flat_shortcut_preserves_extremes_unsigned_history_and_row_boundaries() {
    for header in headers() {
        for width in [1, 2, 3, 17] {
            for y in [0, 1, 2, 5, usize::MAX] {
                for x in 0..width {
                    let mut initial = State::new(header, width).unwrap();
                    initial.next = Some((x, y));
                    // Stale signed margins and unrelated history must not block
                    // the shortcut, including the nonexistent W at x == 0.
                    initial.error.fill(i32::MIN);
                    for pos in signed_positions(&initial, x, y).into_iter().flatten() {
                        initial.error[pos] = 0;
                    }
                    for (i, errors) in initial.pred_errors.iter_mut().enumerate() {
                        for (p, error) in errors.iter_mut().enumerate() {
                            *error = [0, u32::MAX, u32::MAX - 1, 1, 1 << 31][(p + i) % 5];
                        }
                    }
                    for n in [0, -1, 1, i32::MIN, i32::MAX] {
                        for value in [n, n.wrapping_add(1), !n] {
                            let mut actual = initial.clone();
                            let mut reference = initial.clone();
                            assert_eq!(
                                compare_prediction(&mut actual, &mut reference, x, y, [n; 5]),
                                (n, 0)
                            );
                            assert_eq!(actual.pred, i64::from(n) << EXTRA_BITS);
                            assert_eq!(actual.prediction, [actual.pred; 4]);
                            assert_eq!(actual.error, initial.error);
                            assert_eq!(actual.pred_errors, initial.pred_errors);
                            compare_update(&mut actual, &mut reference, value, x, y);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_nonflat_neighbor_retains_the_full_prediction_and_state() {
    for header in headers() {
        for n in [0i32, -17, i32::MIN, i32::MAX] {
            for changed in 0..5 {
                let mut neighbors = [n; 5];
                neighbors[changed] = n.wrapping_add(1);
                assert!(neighbors.iter().any(|&v| v != neighbors[0]));
                let mut actual = State::new(header, 3).unwrap();
                actual.next = Some((1, 2));
                let mut reference = actual.clone();
                compare_prediction(&mut actual, &mut reference, 1, 2, neighbors);
                compare_update(&mut actual, &mut reference, !n, 1, 2);
            }
        }
    }
}

#[test]
fn every_nonzero_signed_error_blocks_flat_shortcut_including_minimum() {
    for header in headers() {
        for error in [-1, 1, i32::MIN, i32::MAX] {
            for y in [1, 2] {
                for direction in 0..4 {
                    let mut actual = State::new(header, 3).unwrap();
                    actual.next = Some((1, y));
                    let position = signed_positions(&actual, 1, y)[direction].unwrap();
                    actual.error[position] = error;
                    let mut reference = actual.clone();
                    let result = compare_prediction(&mut actual, &mut reference, 1, y, [17; 5]);
                    assert_eq!(result.1, error);
                    compare_update(&mut actual, &mut reference, i32::MIN, 1, y);
                }
            }
        }
    }
}

fn neighbors(samples: &[i32], width: usize, x: usize, y: usize) -> [i32; 5] {
    let pos = y * width + x;
    let w = if x > 0 {
        samples[pos - 1]
    } else if y > 0 {
        samples[pos - width]
    } else {
        0
    };
    let n = if y > 0 { samples[pos - width] } else { w };
    let ne = if y > 0 && x + 1 < width {
        samples[pos - width + 1]
    } else {
        n
    };
    let nw = if y > 0 && x > 0 {
        samples[pos - width - 1]
    } else {
        w
    };
    let nn = if y > 1 { samples[pos - 2 * width] } else { n };
    [n, w, ne, nw, nn]
}

#[test]
fn raster_flat_nonflat_transitions_keep_predictions_and_both_history_rows_identical() {
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
                let (mut flat, mut nonflat) = (0, 0);
                for (p, &value) in samples.iter().enumerate() {
                    let (x, y) = (p % width, p / width);
                    let neighbors = neighbors(&samples, width, x, y);
                    let eligible = neighbors.iter().all(|&v| v == neighbors[0])
                        && signed_positions(&actual, x, y)
                            .into_iter()
                            .flatten()
                            .all(|pos| actual.error[pos] == 0);
                    if eligible {
                        flat += 1;
                    } else {
                        nonflat += 1;
                    }
                    compare_prediction(&mut actual, &mut reference, x, y, neighbors);
                    compare_update(&mut actual, &mut reference, value, x, y);
                }
                assert!(flat > 0);
                if matches!(pattern, 0 | 2) {
                    // The first i32::MIN sample has zero narrowed signed error
                    // but nonzero unsigned predictor error: both stay eligible.
                    assert_eq!(flat, samples.len());
                } else {
                    assert!(nonflat > 0, "width={width}, pattern={pattern}");
                }
                if pattern == 4 {
                    assert!(flat > width * 6, "flat recovery width={width}");
                }
                actual.reset();
                reference.reset();
                assert_eq!(actual, reference);
                assert_eq!(actual, State::new(header, width).unwrap());
            }
        }
    }
}

#[test]
fn flat_shortcut_still_validates_call_order_before_mutating_state() {
    let mut actual = State::new(Header::default(), 2).unwrap();
    let mut reference = actual.clone();
    for (x, y) in [(1, 0), (2, 0), (0, 1), (usize::MAX, usize::MAX)] {
        let before = actual.clone();
        assert!(actual.predict(x, y, 0, 0, 0, 0, 0).is_err());
        assert!(
            reference
                .predict_impl::<false>(x, y, 0, 0, 0, 0, 0)
                .is_err()
        );
        assert_eq!(actual, before);
        assert_eq!(actual, reference);
    }
    compare_prediction(&mut actual, &mut reference, 0, 0, [0; 5]);
    let pending = actual.clone();
    assert!(actual.predict(0, 0, 0, 0, 0, 0, 0).is_err());
    assert!(
        reference
            .predict_impl::<false>(0, 0, 0, 0, 0, 0, 0)
            .is_err()
    );
    assert_eq!(actual, pending);
    assert_eq!(actual, reference);
    compare_update(&mut actual, &mut reference, 123, 0, 0);
    compare_prediction(&mut actual, &mut reference, 1, 0, [123; 5]);
    actual.reset();
    reference.reset();
    assert_eq!(actual, reference);
}
