use super::*;

fn predictors() -> Vec<Predictor> {
    (0..14)
        .filter(|&p| p != 6)
        .map(|p| Predictor::try_from(p).unwrap())
        .collect()
}

fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn bounds(lo: i32, hi: i32) -> Bounds {
    Bounds {
        lo: i64::from(lo),
        hi: i64::from(hi),
    }
}

fn check_plane(pixels: &[i32], width: usize, stride: usize, height: usize) -> u16 {
    let ps = predictors();
    let channel = ChannelResiduals::new(pixels, width, stride, height, &ps, true).unwrap();
    let wide = ChannelResiduals::new(pixels, width, stride, height, &ps, false).unwrap();
    assert_eq!(wide.narrow_mask, 0);
    for p in ps {
        for y in 0..height {
            let mut expected = vec![i64::MIN; width + 3];
            let mut actual = vec![i64::MIN; width + 3];
            residual_row_scalar(pixels, width, stride, y, p, &mut expected).unwrap();
            channel.residual_row(y, p, &mut actual).unwrap();
            assert_eq!(actual, expected, "narrow {p:?} width={width} y={y}");
            actual.fill(i64::MIN);
            wide.residual_row(y, p, &mut actual).unwrap();
            assert_eq!(actual, expected, "wide {p:?} width={width} y={y}");
            #[cfg(target_arch = "x86_64")]
            if channel.narrow_mask & (1 << p as u32) != 0 {
                actual.fill(i64::MIN);
                let row = Row::new(pixels, width, stride, y, p, actual.len()).unwrap();
                row.scalar_range(p, &mut actual, 0, 2);
                // SAFETY: the channel's private proof establishes AVX2 and all
                // lane arithmetic; output and row were validated above.
                let next = unsafe { x86::avx2_narrow_row(&row, p, &mut actual) };
                assert!(next >= 10, "must really execute an eight-lane block");
                row.scalar_range(p, &mut actual, next, width);
                assert_eq!(actual, expected, "direct eight-lane {p:?}");
            }
        }
    }
    channel.narrow_mask
}

#[test]
fn experiment5_narrow_proof_boundaries_are_predictor_specific() {
    for p in predictors() {
        assert!(bounds(-65535, 65535).permits(p), "RCT16 {p:?}");
        assert_eq!(bounds(i32::MIN, i32::MAX).permits(p), p == Predictor::Zero);
    }
    for p in [Predictor::Weighted, Predictor::Best, Predictor::Variable] {
        assert!(!bounds(0, 1).permits(p));
    }
    for p in [Predictor::Left, Predictor::Top, Predictor::Select] {
        assert!(bounds(-1_073_741_824, 1_073_741_823).permits(p));
        assert!(!bounds(-1_073_741_824, 1_073_741_824).permits(p));
        assert!(bounds(i32::MIN, -1).permits(p));
        assert!(!bounds(i32::MIN, 0).permits(p));
    }
    for p in [
        Predictor::Average0,
        Predictor::Average1,
        Predictor::Average2,
        Predictor::Average3,
    ] {
        assert!(bounds(-1_073_741_824, 1_073_741_823).permits(p));
        assert!(!bounds(-1_073_741_825, 0).permits(p));
        assert!(!bounds(0, 1_073_741_824).permits(p));
    }
    assert!(bounds(-715_827_882, 715_827_882).permits(Predictor::Gradient));
    assert!(!bounds(-715_827_883, 715_827_883).permits(Predictor::Gradient));
    assert!(!bounds(1_200_000_000, 1_200_000_001).permits(Predictor::Gradient));
    assert!(bounds(-107_374_181, 107_374_181).permits(Predictor::Average4));
    assert!(!bounds(-107_374_182, 107_374_182).permits(Predictor::Average4));
    assert!(bounds(134_217_727, 134_217_727).permits(Predictor::Average4));
    assert!(!bounds(134_217_728, 134_217_728).permits(Predictor::Average4));
    assert!(bounds(-134_217_728, -134_217_728).permits(Predictor::Average4));
    assert!(!bounds(-134_217_729, -134_217_729).permits(Predictor::Average4));
}

#[test]
fn experiment5_narrow_all_widths_padding_rows_and_unaligned_output() {
    let mut state = 0xb866_b3de_u32;
    for width in 1..=41 {
        for padding in [0, 3] {
            let stride = width + padding;
            let height = 4;
            let mut pixels = vec![i32::MAX; stride * (height - 1) + width];
            for y in 0..height {
                for x in 0..width {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    pixels[y * stride + x] = (state % 131071) as i32 - 65535;
                }
            }
            let mask = check_plane(&pixels, width, stride, height);
            assert_eq!(mask, if avx2() && width >= 12 { 0x3fbf } else { 0 });
            let plan =
                ChannelResiduals::new(&pixels, width, stride, height, &predictors(), true).unwrap();
            for p in predictors() {
                let mut expected = vec![0; width];
                residual_row_scalar(&pixels, width, stride, 3, p, &mut expected).unwrap();
                for offset in 0..4 {
                    let mut storage = vec![i64::MIN; width + 6];
                    plan.residual_row(3, p, &mut storage[offset..offset + width])
                        .unwrap();
                    assert_eq!(&storage[offset..offset + width], expected);
                    assert!(
                        storage[..offset]
                            .iter()
                            .chain(&storage[offset + width..])
                            .all(|&v| v == i64::MIN)
                    );
                }
            }
        }
    }
}

#[test]
fn experiment5_narrow_rounding_ties_and_i32_min_division() {
    for &v in &[
        -1_073_741_824,
        -107_374_181,
        -17,
        -2,
        -1,
        0,
        1,
        17,
        107_374_181,
    ] {
        check_plane(&[v; 48], 16, 16, 3);
    }
    let mut pixels = [0; 48];
    for x in 2..10 {
        pixels.fill(0);
        pixels[32 + x - 1] = -9;
        pixels[16 + x] = -3;
        pixels[16 + x - 1] = -6;
        pixels[32 + x] = 17;
        check_plane(&pixels, 16, 16, 3);
        let plan = ChannelResiduals::new(&pixels, 16, 16, 3, &[Predictor::Select], true).unwrap();
        let mut out = [0; 16];
        plan.residual_row(2, Predictor::Select, &mut out).unwrap();
        assert_eq!(out[x], 20); // Equidistant: select top=-3, not left=-9.
        drop(plan);
        pixels.fill(0);
        pixels[32 + x - 1] = -2;
        pixels[32 + x] = 7;
        let plan = ChannelResiduals::new(&pixels, 16, 16, 3, &[Predictor::Average4], true).unwrap();
        plan.residual_row(2, Predictor::Average4, &mut out).unwrap();
        assert_eq!(out[x], 7); // (-14+8)/16 truncates to zero, not -1.
        check_plane(&pixels, 16, 16, 3);
    }
}

#[test]
fn experiment5_narrow_rct_extrema_and_full_i32_fallback() {
    for max in [255, 65535, i32::MAX] {
        let source: Vec<Vec<i32>> = (0..3)
            .map(|c| {
                (0..64)
                    .map(|i| if i & (1 << c) == 0 { 0 } else { max })
                    .collect()
            })
            .collect();
        for rct in 0..42 {
            let mut transformed = source.clone();
            super::super::transforms::rct_forward(&mut transformed, 0, rct).unwrap();
            for pixels in &transformed {
                let mask = check_plane(pixels, 16, 16, 4);
                if avx2() && max <= 65535 {
                    assert_eq!(mask, 0x3fbf, "RCT={rct} max={max}");
                }
            }
        }
    }
    let pixels: Vec<_> = (0..64)
        .map(|i| [i32::MIN, i32::MAX, -1, 0][i % 4])
        .collect();
    assert_eq!(check_plane(&pixels, 16, 16, 4), if avx2() { 1 } else { 0 });
    // Wide high-offset planes can still use copy predictors, not sum predictors.
    let pixels: Vec<_> = (0..64).map(|i| i32::MAX - i % 17).collect();
    let mask = check_plane(&pixels, 16, 16, 4);
    if avx2() {
        assert_ne!(mask & (1 << Predictor::Left as u32), 0);
        assert_eq!(mask & (1 << Predictor::Average4 as u32), 0);
    }
}

#[test]
fn experiment5_narrow_proves_whole_channel_and_keeps_errors_atomic() {
    let mut pixels = [0; 64];
    pixels[63] = i32::MAX;
    let plan = ChannelResiduals::new(&pixels, 16, 16, 4, &[Predictor::Average4], true).unwrap();
    assert_eq!(
        plan.narrow_mask, 0,
        "last-row outlier invalidates every row's narrow proof"
    );
    let mut output = [123; 16];
    for p in [Predictor::Weighted, Predictor::Best, Predictor::Variable] {
        assert_eq!(
            plan.residual_row(0, p, &mut output),
            Err(RowError::UnsupportedPredictor(p))
        );
        assert_eq!(output, [123; 16]);
    }
    assert_eq!(
        plan.residual_row(4, Predictor::Left, &mut output),
        Err(RowError::Plane(PredictorError::PositionOutOfBounds))
    );
    assert_eq!(
        plan.residual_row(0, Predictor::Left, &mut output[..15]),
        Err(RowError::OutputTooSmall)
    );
    assert_eq!(output, [123; 16]);
    let only_zero = ChannelResiduals::new(&pixels, 16, 16, 4, &[Predictor::Zero], true).unwrap();
    assert_eq!(only_zero.narrow_mask & (1 << Predictor::Left as u32), 0);
    // No unchecked proof can be constructed with a missing final row or overflow.
    for (w, stride, h) in [
        (0, 0, 1),
        (2, 1, 1),
        (16, 16, 5),
        (1, usize::MAX, 3),
        (1, 1, 0),
    ] {
        assert!(ChannelResiduals::new(&pixels, w, stride, h, &predictors(), true).is_err());
    }
}
