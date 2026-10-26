use super::*;

fn view(pixels: &[i32], width: usize, height: usize, stride: usize) -> ChannelView<'_> {
    ChannelView {
        pixels,
        width,
        height,
        stride,
    }
}

fn expression(key: RctCostKey, [r, g, b]: [i32; 3]) -> i32 {
    use RctCostKey::*;
    match key {
        R => r,
        G => g,
        B => b,
        RMinusG => r.wrapping_sub(g),
        RMinusB => r.wrapping_sub(b),
        GMinusR => g.wrapping_sub(r),
        GMinusB => g.wrapping_sub(b),
        BMinusR => b.wrapping_sub(r),
        BMinusG => b.wrapping_sub(g),
        RMinusMidGB => r.wrapping_sub(g.wrapping_add(b) >> 1),
        GMinusMidRB => g.wrapping_sub(r.wrapping_add(b) >> 1),
        BMinusMidRG => b.wrapping_sub(r.wrapping_add(g) >> 1),
        YCoCgY | YCoCgCg => {
            let tmp = b.wrapping_add(r.wrapping_sub(b) >> 1);
            let cg = g.wrapping_sub(tmp);
            match key {
                YCoCgY => tmp.wrapping_add(cg >> 1),
                _ => cg,
            }
        }
    }
}

#[test]
fn rct_cost_keys_match_scalar_and_native_planes_with_wrapping() {
    let extremes = [i32::MIN, i32::MAX, -65535, -1, 0, 1, 65535];
    let mut original = [Vec::new(), Vec::new(), Vec::new()];
    for r in extremes {
        for g in extremes {
            for b in extremes {
                for (plane, value) in original.iter_mut().zip([r, g, b]) {
                    plane.push(value);
                }
            }
        }
    }
    let mut random = 0xb866_b3d0u32;
    for _ in 0..258 {
        for plane in &mut original {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            plane.push(random as i32);
        }
    }
    for native in [false, true] {
        let mut seen: [Option<Vec<i32>>; RCT_COST_KEYS] = std::array::from_fn(|_| None);
        for rct_type in RCT_CANDIDATES {
            let mut trial = original.clone();
            if native {
                transforms::rct_forward(&mut trial, 0, rct_type).unwrap();
            } else {
                crate::modular::rct_simd::forward_scalar(&mut trial, 0, rct_type).unwrap();
            }
            for (c, key) in rct_cost_keys(rct_type).unwrap().into_iter().enumerate() {
                for (x, &actual) in trial[c].iter().enumerate() {
                    let input = [original[0][x], original[1][x], original[2][x]];
                    assert_eq!(
                        actual,
                        expression(key, input),
                        "RCT {rct_type}, {key:?}, native={native}, input={input:?}",
                    );
                }
                if let Some(previous) = &seen[key as usize] {
                    assert_eq!(previous, &trial[c], "RCT {rct_type}, {key:?}");
                } else {
                    seen[key as usize] = Some(trial[c].clone());
                }
            }
        }
        assert_eq!(seen.iter().filter(|cost| cost.is_some()).count(), 14);
    }
    let overflow = [i32::MAX, 0, i32::MAX];
    assert_ne!(
        expression(RctCostKey::YCoCgCg, overflow),
        expression(RctCostKey::GMinusMidRB, overflow),
    );
}

#[test]
fn rct_cost_keys_leave_unmapped_candidates_uncached() {
    for rct_type in 0..42 {
        assert_eq!(
            rct_cost_keys(rct_type).is_some(),
            RCT_CANDIDATES.contains(&rct_type),
        );
    }
    assert!(rct_cost_keys(u32::MAX).is_none());
}

fn plane(width: usize, height: usize, padding: usize, wide: bool, mut random: u32) -> Vec<i32> {
    let stride = width + padding;
    let mut pixels = vec![0x5a5a_5a5a; (height - 1) * stride + width];
    let extremes = [i32::MIN, i32::MAX, -1, 0, 1];
    for y in 0..height {
        for x in 0..width {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            pixels[y * stride + x] = if wide {
                if (x + y) % 2 == 0 {
                    extremes[(x + 3 * y) % extremes.len()]
                } else {
                    random as i32
                }
            } else {
                (random & 65535) as i32 - 32768
            };
        }
    }
    pixels
}

fn reference(channels: &[ChannelView<'_>], first_color: usize, mode: Backend) -> Vec<(u32, f32)> {
    let width = channels[first_color].width;
    let height = channels[first_color].height;
    let original: Vec<Vec<i32>> = channels[first_color..first_color + 3]
        .iter()
        .map(|ch| {
            (0..height)
                .flat_map(|y| ch.row(y).iter().copied())
                .collect()
        })
        .collect();
    RCT_CANDIDATES
        .iter()
        .map(|&rct_type| {
            let mut trial = original.clone();
            crate::modular::rct_simd::forward_scalar(&mut trial, 0, rct_type).unwrap();
            let costs: Vec<_> = channels
                .iter()
                .enumerate()
                .map(|(c, &ch)| {
                    let ch = if (first_color..first_color + 3).contains(&c) {
                        view(&trial[c - first_color], width, height, width)
                    } else {
                        ch
                    };
                    channel_cost(ch, mode).unwrap()
                })
                .collect();
            (rct_type, combine_costs(&costs).unwrap())
        })
        .collect()
}

fn assert_selection(actual: &RctSelection, expected: &[(u32, f32)]) {
    assert_eq!(actual.candidates.len(), expected.len());
    let mut winner = (0, f32::MAX);
    for (&(rct, cost), &(expected_rct, expected_cost)) in actual.candidates.iter().zip(expected) {
        assert_eq!(rct, expected_rct);
        assert_eq!(cost.to_bits(), expected_cost.to_bits(), "RCT {rct}");
        if expected_cost < winner.1 {
            winner = (expected_rct, expected_cost);
        }
    }
    assert_eq!(actual.rct_type, winner.0);
    assert_eq!(actual.cost.to_bits(), winner.1.to_bits());
}

#[test]
fn cached_rct_preserves_every_prefix_cost_across_backends_and_strides() {
    let metadata = [i32::MIN, 13, i32::MAX, -3, 0, 17];
    let alpha = [13, -9, 11, -2, 8, 5];
    for (width, height, wide) in [
        (1, 1, false),
        (3, 2, true),
        (7, 3, false),
        (8, 3, true),
        (9, 3, false),
        (17, 5, true),
    ] {
        let originals: Vec<_> = (0..3)
            .map(|c| plane(width, height, c + 1, wide, 0xb866_b3d0 + c as u32))
            .collect();
        let before = originals.clone();
        let channels = [
            view(&metadata, 3, 2, 3),
            view(&originals[0], width, height, width + 1),
            view(&originals[1], width, height, width + 2),
            view(&originals[2], width, height, width + 3),
            view(&alpha, 2, 3, 2),
        ];
        for mode in super::tests::supported() {
            let expected = reference(&channels, 1, mode);
            for count in 1..=RCT_CANDIDATES.len() {
                let actual =
                    select_rct_with_cost(&channels, 1, count, |ch| channel_cost(ch, mode)).unwrap();
                assert_selection(&actual, &expected[..count]);
            }
            if mode == backend() {
                assert_selection(&select_rct(&channels, 1, 19).unwrap(), &expected);
            }
        }
        assert_eq!(originals, before);
    }
}

#[test]
fn cached_rct_scans_fourteen_color_expressions_and_other_channels_once() {
    let pixels = plane(9, 5, 2, true, 0xb866_b3d0);
    let metadata = [0, 1, 2, 3, 4, 5];
    let alpha = [9, 8, 7];
    let channels = [
        view(&metadata, 3, 2, 3),
        view(&pixels, 9, 5, 11),
        view(&pixels, 9, 5, 11),
        view(&pixels, 9, 5, 11),
        view(&alpha, 3, 1, 3),
    ];
    let color_scans = [
        3, 6, 8, 10, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    ];
    for (prefix, expected) in color_scans.into_iter().enumerate() {
        let mut scans = [0; 3];
        select_rct_with_cost(&channels, 1, prefix + 1, |ch| {
            let index = match (ch.width, ch.height) {
                (3, 2) => 0,
                (3, 1) => 1,
                (9, 5) => 2,
                _ => panic!("unexpected scored shape"),
            };
            scans[index] += 1;
            channel_cost(ch, Backend::Scalar)
        })
        .unwrap();
        assert_eq!(scans, [1, 1, expected], "prefix {}", prefix + 1);
    }
}

#[test]
fn cached_rct_keeps_first_tie_for_zero_and_empty_planes() {
    let zeros = [0; 33];
    for ch in [
        view(&zeros, 11, 3, 11),
        view(&[], 0, usize::MAX, usize::MAX),
        view(&[], usize::MAX, 0, usize::MAX),
    ] {
        for count in 1..=RCT_CANDIDATES.len() {
            let actual = select_rct(&[ch; 3], 0, count).unwrap();
            assert_eq!(actual.rct_type, 0);
            assert_eq!(actual.cost.to_bits(), 0.0f32.to_bits());
            let expected: Vec<_> = RCT_CANDIDATES[..count]
                .iter()
                .map(|&rct| (rct, 0.0))
                .collect();
            assert_selection(&actual, &expected);
        }
    }
}

#[test]
fn cached_rct_validates_before_scoring_and_does_not_retain_failed_work() {
    let pixels = plane(9, 5, 0, true, 0xb866_b3d0);
    let channels = [view(&pixels, 9, 5, 9); 3];
    for (first_color, count) in [(0, 0), (0, 20), (usize::MAX, 19), (1, 19)] {
        let mut calls = 0;
        assert!(
            select_rct_with_cost(&channels, first_color, count, |ch| {
                calls += 1;
                channel_cost(ch, Backend::Scalar)
            })
            .is_err()
        );
        assert_eq!(calls, 0);
    }
    let mut calls = 0;
    let failed = select_rct_with_cost(&channels, 0, 19, |ch| {
        calls += 1;
        if calls == 4 {
            Err(Error::Limit("injected RCT cost failure"))
        } else {
            channel_cost(ch, Backend::Scalar)
        }
    });
    assert!(matches!(
        failed,
        Err(Error::Limit("injected RCT cost failure"))
    ));
    assert_eq!(calls, 4);
    calls = 0;
    let actual = select_rct_with_cost(&channels, 0, 19, |ch| {
        calls += 1;
        channel_cost(ch, Backend::Scalar)
    })
    .unwrap();
    assert_eq!(calls, 14);
    assert_selection(&actual, &reference(&channels, 0, Backend::Scalar));
}
