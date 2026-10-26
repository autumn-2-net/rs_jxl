use super::*;

fn view(pixels: &[i32], width: usize, height: usize, stride: usize) -> ChannelView<'_> {
    ChannelView {
        pixels,
        width,
        height,
        stride,
    }
}

fn plane(width: usize, height: usize, padding: usize, mut random: u32) -> Vec<i32> {
    let stride = width + padding;
    let mut pixels = vec![i32::MIN; (height - 1) * stride + width];
    for y in 0..height {
        for x in 0..width {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            pixels[y * stride + x] = (random & 65535) as i32 - 32768;
        }
    }
    pixels
}

fn assert_winner(actual: &WeightedPresetSelection, expected: &WeightedPresetSelection) {
    assert_eq!(actual.preset, expected.preset);
    assert_eq!(actual.cost.to_bits(), expected.cost.to_bits());
    let mut work = WpCostWork::default();
    for candidate in &actual.candidates {
        work.add(candidate.work).unwrap();
    }
    assert_eq!(actual.work, work);
}

fn assert_identical(actual: &WeightedPresetSelection, expected: &WeightedPresetSelection) {
    assert_winner(actual, expected);
    assert_eq!(actual.work, expected.work);
    assert_eq!(actual.candidates.len(), expected.candidates.len());
    for (actual, expected) in actual.candidates.iter().zip(&expected.candidates) {
        assert_eq!(actual.cost.to_bits(), expected.cost.to_bits());
        assert_eq!(actual.unhealthy, expected.unhealthy);
        assert_eq!(actual.work, expected.work);
    }
}

#[test]
fn parallel_wp_presets_match_every_unbounded_mode_with_padded_independent_channels() {
    let a = plane(257, 128, 3, 0xb866_b3d0);
    let b = plane(255, 128, 2, 0x1632_77bd);
    let channels = [
        view(&a, 257, 128, 260),
        view(&[], 0, usize::MAX, 0),
        view(&b, 255, 128, 257),
    ];
    assert_eq!(validate(&channels).unwrap(), WP_PARALLEL_MIN_PIXELS);
    for count in [3, 5] {
        let expected = select_weighted_preset(&channels, count, false).unwrap();
        for threads in [2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let actual = pool.install(|| {
                assert!(parallel_weighted_presets(WP_PARALLEL_MIN_PIXELS, count));
                select_weighted_preset_parallel(&channels, count, false).unwrap()
            });
            assert_identical(&actual, &expected);
        }
    }
}

#[test]
fn parallel_wp_presets_fixed_bound_preserves_serial_winner_and_completed_costs() {
    let pixels = plane(256, 256, 1, 0x8e41_5a17);
    let channels = [view(&pixels, 256, 256, 257)];
    let unbounded = select_weighted_preset(&channels, 5, false).unwrap();
    let serial = select_weighted_preset(&channels, 5, true).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let actual = pool.install(|| select_weighted_preset_parallel(&channels, 5, true).unwrap());
    assert_winner(&actual, &serial);
    assert_winner(&actual, &unbounded);
    assert_eq!(actual.candidates.len(), 5);
    assert_eq!(actual.candidates[0].work.checkpoints, 0);
    for (candidate, full) in actual.candidates.iter().zip(&unbounded.candidates) {
        if candidate.work.pruned_modes == 0 {
            assert_eq!(candidate.cost.to_bits(), full.cost.to_bits());
            assert_eq!(candidate.unhealthy, full.unhealthy);
        } else {
            assert_eq!(candidate.cost, f32::MAX);
        }
    }
    assert!(actual.work.checkpoints > 0);
}

#[test]
fn parallel_wp_presets_keep_first_mode_for_ties_and_all_unhealthy_candidates() {
    let pixels = vec![0; WP_PARALLEL_MIN_PIXELS];
    let flat = [view(&pixels, 256, 256, 256)];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let serial = select_weighted_preset(&flat, 5, true).unwrap();
    let actual = pool.install(|| select_weighted_preset_parallel(&flat, 5, true).unwrap());
    assert_identical(&actual, &serial);
    assert_eq!((actual.preset, actual.cost), (0, 0.0));
    assert!(
        actual
            .candidates
            .iter()
            .all(|c| c.cost == 0.0 && !c.unhealthy)
    );

    let unhealthy = [view(&[1, i32::MIN], 2, 1, 2), flat[0]];
    for bounded in [false, true] {
        let serial = select_weighted_preset(&unhealthy, 5, bounded).unwrap();
        let actual =
            pool.install(|| select_weighted_preset_parallel(&unhealthy, 5, bounded).unwrap());
        assert_identical(&actual, &serial);
        assert_eq!((actual.preset, actual.cost), (0, f32::MAX));
        assert_eq!(actual.work.pixels, 10);
        assert_eq!(actual.work.checkpoints, 0);
        assert!(actual.candidates.iter().all(|c| c.unhealthy));
    }
}

#[test]
fn parallel_wp_presets_unhealthy_mode_zero_leaves_a_later_healthy_mode_unbounded() {
    // The ordinary preset predicts 12 at (1, 1); mode 1 predicts 11.
    // Only the latter can subtract i32::MIN + 11 without overflow. Complete
    // the channel with its own predictions, so it stays healthy afterwards.
    let mut prefix = [-7, 5, 12, -1, 6, i32::MIN + 11, 0, 0];
    let mut state = weighted::State::new(weighted::Header::preset(1).unwrap(), 4).unwrap();
    for p in 0..prefix.len() {
        let (x, y) = (p % 4, p / 4);
        let n = view(&prefix, 4, 2, 4).neighbors(x, y);
        let (guess, _) = state
            .predict(x, y, n.top, n.left, n.topright, n.topleft, n.toptop)
            .unwrap();
        if p == 5 {
            assert_eq!(guess, 11);
        }
        if p >= 6 {
            prefix[p] = guess;
        }
        state.update(prefix[p], x, y).unwrap();
    }
    let pixels = vec![0; WP_PARALLEL_MIN_PIXELS];
    let channels = [view(&prefix, 4, 2, 4), view(&pixels, 256, 256, 256)];
    let serial = select_weighted_preset(&channels, 5, true).unwrap();
    assert!(serial.candidates[0].unhealthy);
    assert!(!serial.candidates[1].unhealthy);
    assert!(serial.cost < f32::MAX);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let actual = pool.install(|| select_weighted_preset_parallel(&channels, 5, true).unwrap());
    assert_winner(&actual, &serial);
    assert!(actual.candidates.iter().all(|c| c.work.checkpoints == 0));
    assert_eq!(
        actual.candidates[1].cost.to_bits(),
        serial.candidates[1].cost.to_bits()
    );
}

#[test]
fn parallel_wp_presets_respect_null_serial_small_and_single_worker_gates() {
    let pixels = vec![0; WP_PARALLEL_MIN_PIXELS];
    let channels = [view(&pixels, 256, 256, 256)];
    let expected = select_weighted_preset(&channels, 3, true).unwrap();
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                assert!(rayon::current_thread_index().is_none());
                assert!(!parallel_weighted_presets(WP_PARALLEL_MIN_PIXELS, 3));
                let actual = select_weighted_preset_parallel(&channels, 3, true).unwrap();
                assert_identical(&actual, &expected);
                assert!(rayon::current_thread_index().is_none());
            })
            .join()
            .unwrap();
    });
    for threads in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(
                parallel_weighted_presets(WP_PARALLEL_MIN_PIXELS, 3),
                threads > 1
            );
            assert!(!parallel_weighted_presets(WP_PARALLEL_MIN_PIXELS - 1, 5));
            for count in [1, 2] {
                assert!(!parallel_weighted_presets(WP_PARALLEL_MIN_PIXELS, count));
                let serial = select_weighted_preset(&channels, count, true).unwrap();
                let actual = select_weighted_preset_parallel(&channels, count, true).unwrap();
                assert_identical(&actual, &serial);
            }
            if threads == 1 {
                let actual = select_weighted_preset_parallel(&channels, 3, true).unwrap();
                assert_identical(&actual, &expected);
            }
            crate::execution::serial(|| {
                assert!(!parallel_weighted_presets(WP_PARALLEL_MIN_PIXELS, 3));
                let actual = select_weighted_preset_parallel(&channels, 3, true).unwrap();
                assert_identical(&actual, &expected);
            });
            let small = [view(&pixels[..65535], 255, 257, 255)];
            let serial = select_weighted_preset(&small, 3, true).unwrap();
            let actual = select_weighted_preset_parallel(&small, 3, true).unwrap();
            assert_identical(&actual, &serial);
        });
    }
}

#[test]
fn parallel_wp_presets_preserve_empty_inputs_and_validation_errors() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    pool.install(|| {
        for channels in [vec![], vec![view(&[], 0, usize::MAX, 0)]] {
            for count in [1, 3, 5] {
                let expected = select_weighted_preset(&channels, count, true).unwrap();
                let actual = select_weighted_preset_parallel(&channels, count, true).unwrap();
                assert_identical(&actual, &expected);
            }
        }
        for count in [0, 6, usize::MAX] {
            let expected = select_weighted_preset(&[], count, true).unwrap_err();
            let actual = select_weighted_preset_parallel(&[], count, true).unwrap_err();
            assert_eq!(actual.to_string(), expected.to_string());
        }
        for channel in [
            view(&[0; 3], 2, 2, 2),
            view(&[0; 4], 2, 2, 1),
            view(&[], usize::MAX, 2, usize::MAX),
            view(&[], i32::MAX as usize + 1, 1, i32::MAX as usize + 1),
        ] {
            for count in [0, 5] {
                let expected = select_weighted_preset(&[channel], count, true).unwrap_err();
                let actual = select_weighted_preset_parallel(&[channel], count, true).unwrap_err();
                assert_eq!(actual.to_string(), expected.to_string());
            }
        }
    });
}
