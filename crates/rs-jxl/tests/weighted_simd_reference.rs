//! Primitive arithmetic equality, not an encoded-byte acceptance requirement.

#[test]
fn four_lane_error_history_matches_wrapping_scalar_arithmetic() {
    let words = [0, 1, -1, i32::MIN, i32::MAX, 0x12345678, -981273];
    for &a in &words {
        for &b in &words {
            for &c in &words {
                let n = [a, b, c, a.wrapping_add(b)];
                let ne = [b, c, a, b.wrapping_sub(c)];
                let nw = [c, a, b, c.wrapping_add(a)];
                let expected =
                    std::array::from_fn(|i| n[i].wrapping_add(ne[i]).wrapping_add(nw[i]));
                assert_eq!(jxl_simd::wp::sum3(&n, &ne, &nw), expected);
                assert_eq!(
                    jxl_simd::wp::sum3_u32(
                        &n.map(|v| v as u32),
                        &ne.map(|v| v as u32),
                        &nw.map(|v| v as u32)
                    ),
                    expected.map(|v| v as u32)
                );
                let mut actual = n;
                jxl_simd::wp::accumulate(&mut actual, &ne);
                assert_eq!(actual, std::array::from_fn(|i| n[i].wrapping_add(ne[i])));
                let mut unsigned = n.map(|v| v as u32);
                jxl_simd::wp::accumulate_u32(&mut unsigned, &ne.map(|v| v as u32));
                assert_eq!(unsigned, actual.map(|v| v as u32));
            }
        }
    }
}

#[test]
fn four_lane_residual_errors_match_wide_reference_inside_envelope() {
    let limit = (1 << 29) - 1;
    for value in [-limit, -7, 0, 7, limit] {
        for seed in [-limit, -1, 0, 1, limit] {
            let predictions = [seed, -seed, seed / 2, seed / 3];
            let expected =
                predictions.map(|p| (((i64::from(p) - i64::from(value)).abs() + 3) >> 3) as i32);
            assert_eq!(jxl_simd::wp::err4(&predictions, value, 3, 3), expected);
        }
    }
}
