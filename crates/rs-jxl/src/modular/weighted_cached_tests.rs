use super::tests::{full_history_reference, neighbors};
use super::*;

fn headers() -> Vec<Header> {
    let mut headers: Vec<_> = (0..5).map(|mode| Header::preset(mode).unwrap()).collect();
    headers.extend((0..16).map(|weight| Header {
        w: [weight, 15 - weight, weight ^ 5, weight ^ 10],
        ..Header::default()
    }));
    headers.push(Header {
        p1_c: 31,
        p2_c: 31,
        p3_ca: 31,
        p3_cb: 31,
        p3_cc: 31,
        p3_cd: 31,
        p3_ce: 31,
        w: [15; 4],
    });
    headers
}

// Retain the pre-cache lookup expression as a differential oracle.
fn uncached_weights(errors: [u32; 4], max_weights: [u32; 4]) -> [u32; 4] {
    if (errors[0] | errors[1] | errors[2] | errors[3]) < 256 {
        std::array::from_fn(|i| SMALL_WEIGHTS[max_weights[i] as usize][errors[i] as usize])
    } else {
        std::array::from_fn(|i| error_weight(errors[i], max_weights[i]))
    }
}

#[test]
fn cached_rows_match_uncached_weights_for_every_small_index_and_weight() {
    for header in headers() {
        let mut state = State::new(header, 3).unwrap();
        for lane in 0..NUM_PREDICTORS {
            assert!(std::ptr::eq(
                state.small_weights[lane],
                &SMALL_WEIGHTS[header.w[lane] as usize]
            ));
        }
        for error in 0..256 {
            let errors = [error, 255 - error, error ^ 85, error ^ 170];
            assert_eq!(
                weights(errors, header.w, state.small_weights),
                uncached_weights(errors, header.w),
                "header={header:?}, errors={errors:?}"
            );
        }
        let tables = state.small_weights;
        state.predict_and_update_raster::<false>(123, 0, 0, 0, 0, 0, 0, 0);
        state.reset();
        for (actual, original) in state.small_weights.into_iter().zip(tables) {
            assert!(std::ptr::eq(actual, original));
        }
        assert_eq!(state, State::new(header, 3).unwrap());
    }
}

#[test]
fn cached_weights_keep_large_errors_and_wrapped_neighborhoods_exact() {
    let boundaries = [
        0,
        1,
        254,
        255,
        256,
        257,
        1 << 16,
        1 << 31,
        u32::MAX - 1,
        u32::MAX,
    ];
    for header in headers() {
        let state = State::new(header, 1).unwrap();
        for a in boundaries {
            for b in boundaries {
                for c in boundaries {
                    let sum = a.wrapping_add(b).wrapping_add(c);
                    for lane in 0..NUM_PREDICTORS {
                        // Only one lane crosses the fast-path boundary, or a
                        // wide unsigned sum wraps back into the small table.
                        let mut errors = [0, 255, 1, 254];
                        errors[lane] = sum;
                        assert_eq!(
                            weights(errors, header.w, state.small_weights),
                            uncached_weights(errors, header.w),
                            "header={header:?}, errors={errors:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cached_weight_rasters_match_independent_wide_full_history() {
    for width in [1, 2, 3, 7, 16, 33] {
        for pattern in 0..7 {
            let mut random = 0x8e41_5a17u32;
            let samples: Vec<i32> = (0..width * 11)
                .map(|pos| {
                    random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                    match pattern {
                        0 => 0,
                        1 => -17,
                        2 => pos as i32 - 127,
                        3 => (random & 255) as i32 - 128,
                        4 => [i32::MIN, i32::MAX, 0][pos % 3],
                        5 => {
                            if pos == width * 5 + width / 2 {
                                i32::MAX
                            } else {
                                0
                            }
                        }
                        _ => random as i32,
                    }
                })
                .collect();
            for header in headers() {
                let expected = full_history_reference(&samples, width, header);
                let mut raster = State::new(header, width).unwrap();
                let mut interior = raster.clone();
                for (pos, &value) in samples.iter().enumerate() {
                    let (x, y) = (pos % width, pos / width);
                    let [n, w, ne, nw, nn] = neighbors(&samples, width, x, y);
                    let actual =
                        raster.predict_and_update_raster::<false>(value, x, y, n, w, ne, nw, nn);
                    let specialized = if x > 0 && x + 1 < width {
                        interior.predict_and_update_raster_interior::<false>(
                            value, x, y, n, w, ne, nw, nn,
                        )
                    } else {
                        interior.predict_and_update_raster::<false>(value, x, y, n, w, ne, nw, nn)
                    };
                    assert_eq!(
                        actual, expected[pos],
                        "width={width}, pattern={pattern}, pos={pos}, header={header:?}"
                    );
                    assert_eq!(specialized, actual);
                    assert_eq!(interior, raster, "complete raster state at ({x}, {y})");
                }
            }
        }
    }
}
