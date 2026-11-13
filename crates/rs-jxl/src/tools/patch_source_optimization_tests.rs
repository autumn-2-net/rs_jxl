// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! Test-only Source Patch optimization oracles, frozen from 1029f1a.
//! Compare detector keys and positions, not encoded JXL bytes.

use super::*;

mod helper_contract {
    use super::*;

    fn tolerance(bits: u32) -> i32 {
        match bits {
            1..=13 => 0,
            14 => 1,
            15 => 3,
            16 => 6,
            _ => unreachable!(),
        }
    }

    #[test]
    fn every_integer_value_matches_scalar_at_threshold_neighbors() {
        for bits in 1..=16 {
            let maximum = (1i32 << bits) - 1;
            let planes = vec![(0..=maximum).collect::<Vec<_>>()];
            let view = DetectionView::integer(ImageView {
                width: planes[0].len(),
                height: 1,
                bit_depth: bits,
                color_channels: 1,
                planes: &planes,
            });
            let threshold = tolerance(bits);
            // Every valid base, both sides of the threshold, and equality.
            // At depth 16 this is under one million pairs.
            for a in 0..=maximum {
                for delta in -threshold - 1..=threshold + 1 {
                    let b = a + delta;
                    if !(0..=maximum).contains(&b) {
                        continue;
                    }
                    let expected = scalar_same(view.pick(a as usize), view.pick(b as usize));
                    assert_eq!(
                        expected,
                        delta.abs() <= threshold,
                        "bits={bits} a={a} b={b}"
                    );
                    assert_eq!(
                        view.same_at(a as usize, b as usize),
                        expected,
                        "bits={bits} a={a} b={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn integer_rgb_checks_every_color_channel_but_not_extras() {
        for bits in 1..=16 {
            let maximum = (1i32 << bits) - 1;
            let threshold = tolerance(bits);
            for channel in 0..3 {
                for base in [0, maximum / 2, maximum] {
                    for delta in [-threshold - 1, -threshold, 0, threshold, threshold + 1] {
                        let value = base + delta;
                        if !(0..=maximum).contains(&value) {
                            continue;
                        }
                        let mut planes = vec![vec![base, base]; 3];
                        planes[channel][1] = value;
                        planes.push(vec![0, maximum]);
                        let view = DetectionView::integer(ImageView {
                            width: 2,
                            height: 1,
                            bit_depth: bits,
                            color_channels: 3,
                            planes: &planes,
                        });
                        let expected = scalar_same(view.pick(0), view.pick(1));
                        assert_eq!(
                            view.same_at(0, 1),
                            expected,
                            "bits={bits} channel={channel} base={base} delta={delta}"
                        );
                        assert_eq!(view.same_at(1, 0), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn pick_channel_preserves_every_integer_scalar_normalization_bit() {
        for bits in 1..=16 {
            let maximum = (1i32 << bits) - 1;
            let scale = 1.0f32 / maximum as f32;
            for colors in [1, 3] {
                let mut planes = vec![(0..=maximum).collect::<Vec<_>>()];
                if colors == 3 {
                    planes.push((0..=maximum).rev().collect());
                    planes.push((0..=maximum).map(|v| (v * 17) & maximum).collect());
                }
                planes.push(vec![maximum; maximum as usize + 1]);
                let view = DetectionView::integer(ImageView {
                    width: planes[0].len(),
                    height: 1,
                    bit_depth: bits,
                    color_channels: colors,
                    planes: &planes,
                });
                for position in 0..planes[0].len() {
                    for channel in 0..3 {
                        let plane = if colors == 1 { 0 } else { channel };
                        let expected = (planes[plane][position] as f32 * scale).to_bits();
                        assert_eq!(
                            view.pick_channel(position, channel).to_bits(),
                            expected,
                            "bits={bits} colors={colors} position={position} channel={channel}"
                        );
                        assert_eq!(
                            view.pick_channel(position, channel).to_bits(),
                            view.pick(position)[channel].to_bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn float_helpers_preserve_signed_zero_nonfinite_payloads_and_scalar_comparison() {
        let values = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            1e-4,
            f32::from_bits(1e-4f32.to_bits() + 1),
            -1.0,
            1.0,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_5678),
            f32::from_bits(0x7f80_0001),
        ];
        for colors in [1, 3] {
            let mut planes = vec![values.to_vec()];
            if colors == 3 {
                planes.push(values.iter().copied().rev().collect());
                planes.push(
                    (0..values.len())
                        .map(|i| values[(i + 3) % values.len()])
                        .collect(),
                );
            }
            planes.push(vec![f32::NAN; values.len()]);
            let view = DetectionView {
                width: values.len(),
                height: 1,
                color_channels: colors,
                samples: DetectionSamples::Float(&planes),
            };
            for a in 0..values.len() {
                for channel in 0..3 {
                    let plane = if colors == 1 { 0 } else { channel };
                    assert_eq!(
                        view.pick_channel(a, channel).to_bits(),
                        planes[plane][a].to_bits()
                    );
                    assert_eq!(
                        view.pick_channel(a, channel).to_bits(),
                        view.pick(a)[channel].to_bits()
                    );
                }
                for b in 0..values.len() {
                    assert_eq!(
                        view.same_at(a, b),
                        scalar_same(view.pick(a), view.pick(b)),
                        "colors={colors} a={a} b={b}"
                    );
                }
            }
        }
    }
}

mod detector_equivalence {
    use super::super::super::ReferenceImage;
    use super::*;

    fn blank(bits: u32, colors: usize) -> ReferenceImage {
        let (width, height) = (97, 69);
        let maximum = (1 << bits) - 1;
        let mut planes = vec![vec![0; width * height]; colors + 2];
        for (i, value) in planes[colors + 1].iter_mut().enumerate() {
            *value = (i as i32 * 17 + 31) & maximum;
        }
        ReferenceImage {
            width,
            height,
            bit_depth: bits,
            color_channels: colors,
            planes,
        }
    }

    fn rectangle(image: &mut ReferenceImage, x: usize, y: usize, w: usize, h: usize, value: i32) {
        for plane in &mut image.planes[..image.color_channels] {
            for row in y..y + h {
                plane[row * image.width + x..row * image.width + x + w].fill(value);
            }
        }
    }

    fn compare(view: DetectionView<'_>, config: &SourcePatchConfig) -> Vec<PatchInfo> {
        let old_background = legacy::legacy_background(view, config, &mut Work::default()).unwrap();
        let new_background = background(view, config, &mut Work::default()).unwrap();
        assert_eq!(
            new_background, old_background,
            "background origins fast={}",
            config.fast
        );
        let old = legacy::legacy_detect_view(view, config, &mut Work::default()).unwrap();
        let mut work = Work::default();
        let new = detect_view(view, config, &mut work).unwrap();
        assert_eq!(
            work.delta_buffers,
            new.len(),
            "one buffer per retained group"
        );
        assert_eq!(new.len(), old.len(), "group count fast={}", config.fast);
        for (index, (actual, expected)) in new.iter().zip(&old).enumerate() {
            assert_eq!(
                actual.key, expected.key,
                "group {index} key fast={}",
                config.fast
            );
            assert_eq!(
                actual.positions, expected.positions,
                "group {index} positions fast={}",
                config.fast
            );
        }
        new
    }

    #[test]
    fn all_integer_depths_gray_rgb_extras_fast_peak_and_divisor() {
        for bits in 1..=16 {
            for colors in [1, 3] {
                let mut image = blank(bits, colors);
                let maximum = (1 << bits) - 1;
                let tolerance = match bits {
                    14 => 1,
                    15 => 3,
                    16 => 6,
                    _ => 0,
                };
                for (c, plane) in image.planes[..colors].iter_mut().enumerate() {
                    for (i, value) in plane.iter_mut().enumerate() {
                        *value = ((i * 13 + c * 7) % (tolerance + 1)) as i32;
                    }
                }
                rectangle(&mut image, 13, 17, 5, 5, maximum);
                rectangle(&mut image, 65, 37, 5, 5, maximum);
                rectangle(&mut image, 41, 49, 3, 7, maximum);
                rectangle(&mut image, 0, 0, 4, 5, maximum);
                rectangle(&mut image, 93, 64, 4, 5, maximum);
                // Distinct RGB deltas exercise channel-major fingerprint order.
                if colors == 3 {
                    for &(x, y, w, h) in &[(13, 17, 5, 5), (65, 37, 5, 5)] {
                        for row in y..y + h {
                            image.planes[1][row * image.width + x..row * image.width + x + w]
                                .fill(maximum / 2);
                        }
                    }
                }
                for fast in [false, true] {
                    for min_peak in [1, 2] {
                        for fingerprint_divisor in [1, 2, 4] {
                            let config = SourcePatchConfig {
                                fast,
                                min_peak,
                                fingerprint_divisor,
                                ..SourcePatchConfig::default()
                            };
                            let result = compare(DetectionView::integer(image.view()), &config);
                            assert!(!result.is_empty(), "bits={bits} colors={colors}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn quantized_collision_keeps_xy_representative_exact_deltas() {
        for colors in [1, 3] {
            for (a, b, min_peak, divisor, grouped) in [
                (64, 44, 2, 1, false),
                (64, 44, 2, 2, true),
                (64, 44, 2, 4, true),
                (30, 31, 1, 1, true),
                (30, 31, 2, 1, false),
            ] {
                let mut image = blank(8, colors);
                // Scan sees the upper-right glyph first; xy representative is lower-left.
                rectangle(&mut image, 65, 13, 5, 5, b);
                rectangle(&mut image, 13, 41, 5, 5, a);
                for fast in [false, true] {
                    let config = SourcePatchConfig {
                        fast,
                        min_peak,
                        fingerprint_divisor: divisor,
                        ..SourcePatchConfig::default()
                    };
                    let result = compare(DetectionView::integer(image.view()), &config);
                    if grouped {
                        assert_eq!(result.len(), 1);
                        assert_eq!(result[0].positions, [(13, 41), (65, 13)]);
                        assert_eq!(result[0].key.samples, vec![a; 25 * colors]);
                    } else {
                        assert!(result.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn rejected_singleton_oversized_low_peak_and_limited_components() {
        let mut image = blank(8, 3);
        rectangle(&mut image, 13, 17, 5, 5, 200);
        rectangle(&mut image, 65, 37, 5, 5, 200);
        rectangle(&mut image, 41, 49, 3, 7, 180); // Singleton.
        rectangle(&mut image, 31, 7, 35, 2, 200); // Oversized.
        rectangle(&mut image, 17, 57, 5, 5, 16); // Low peak, but outside background tolerance.
        for fast in [false, true] {
            for limits in [
                PatchOptions::default(),
                PatchOptions {
                    max_patch_size: 4,
                    ..PatchOptions::default()
                },
                PatchOptions {
                    min_occurrences: 3,
                    ..PatchOptions::default()
                },
                PatchOptions {
                    min_max_patch_area: 26,
                    ..PatchOptions::default()
                },
                PatchOptions {
                    max_positions: 1,
                    ..PatchOptions::default()
                },
                PatchOptions {
                    max_unique_patches: 1,
                    ..PatchOptions::default()
                },
            ] {
                compare(
                    DetectionView::integer(image.view()),
                    &SourcePatchConfig {
                        fast,
                        limits,
                        ..SourcePatchConfig::default()
                    },
                );
            }
        }
        for (width, height) in [(1, 1), (11, 29), (29, 11), (12, 12), (16, 16)] {
            let planes = vec![vec![0; width * height]];
            let view = DetectionView::integer(ImageView {
                width,
                height,
                bit_depth: 16,
                color_channels: 1,
                planes: &planes,
            });
            for fast in [false, true] {
                assert!(
                    compare(
                        view,
                        &SourcePatchConfig {
                            fast,
                            ..SourcePatchConfig::default()
                        }
                    )
                    .is_empty()
                );
            }
        }
    }

    #[test]
    fn normalized_subtraction_boundaries_and_background_relative_collisions() {
        for fast in [false, true] {
            for (right_background, left_value, right_value, min_peak, grouped) in [
                (24, 20, 44, 1, false),
                (24, 20, 45, 1, true),
                (150, 140, 86, 2, true),
            ] {
                let mut image = blank(8, 3);
                let left_background = if right_background == 150 { 200 } else { 0 };
                for plane in &mut image.planes[..3] {
                    for row in plane.chunks_exact_mut(image.width) {
                        row[..48].fill(left_background);
                        row[48..].fill(right_background);
                    }
                }
                rectangle(&mut image, 13, 17, 5, 5, left_value);
                rectangle(&mut image, 65, 37, 5, 5, right_value);
                let result = compare(
                    DetectionView::integer(image.view()),
                    &SourcePatchConfig {
                        fast,
                        min_peak,
                        ..SourcePatchConfig::default()
                    },
                );
                if grouped {
                    assert_eq!(result.len(), 1);
                    assert_eq!(result[0].positions, [(13, 17), (65, 37)]);
                    assert_eq!(
                        result[0].key.samples,
                        vec![left_value - left_background; 75]
                    );
                } else {
                    assert!(result.is_empty());
                }
            }

            let mut image = blank(8, 1);
            for row in image.planes[0].chunks_exact_mut(image.width) {
                row[48..].fill(128);
            }
            // A component spanning incompatible background colors is rejected.
            rectangle(&mut image, 43, 17, 10, 5, 255);
            assert!(
                compare(
                    DetectionView::integer(image.view()),
                    &SourcePatchConfig {
                        fast,
                        ..SourcePatchConfig::default()
                    }
                )
                .is_empty()
            );
        }
    }

    #[test]
    fn float_detector_finite_nonfinite_and_channel_major_fingerprints() {
        for colors in [1, 3] {
            let mut image = blank(8, colors);
            rectangle(&mut image, 65, 13, 5, 5, 44);
            rectangle(&mut image, 13, 41, 5, 5, 64);
            let base: Vec<Vec<f32>> = image
                .planes
                .iter()
                .enumerate()
                .map(|(c, plane)| {
                    plane
                        .iter()
                        .map(|&v| {
                            if c >= colors {
                                f32::NAN
                            } else {
                                v as f32 * (1.0f32 / 255.0)
                            }
                        })
                        .collect()
                })
                .collect();
            for scenario in 0..5 {
                let mut planes = base.clone();
                match scenario {
                    0 => {}
                    1 => {
                        // Finite negative backgrounds and unequal RGB channels.
                        for (c, plane) in planes[..colors].iter_mut().enumerate() {
                            for value in plane {
                                *value -= 0.125 * (c + 1) as f32;
                            }
                        }
                    }
                    2 => {
                        // Nonfinite words inside the repeated candidates reject them.
                        for &(x, y) in &[(65, 13), (13, 41)] {
                            planes[colors - 1][y * image.width + x] = f32::from_bits(0x7fc0_1234);
                        }
                    }
                    3 => {
                        // Isolated infinities and NaN do not erase unrelated patches.
                        planes[0][60 * image.width + 41] = f32::INFINITY;
                        planes[0][60 * image.width + 49] = f32::NEG_INFINITY;
                        planes[0][60 * image.width + 57] = f32::NAN;
                        planes[0][0] = -0.0;
                    }
                    4 => {
                        // No finite seed: infinity is not even same as itself.
                        for plane in &mut planes[..colors] {
                            plane.fill(f32::INFINITY);
                        }
                    }
                    _ => unreachable!(),
                }
                let view = DetectionView {
                    width: image.width,
                    height: image.height,
                    color_channels: colors,
                    samples: DetectionSamples::Float(&planes),
                };
                for fast in [false, true] {
                    for min_peak in [1, 2] {
                        for fingerprint_divisor in [1, 2, 4] {
                            let result = compare(
                                view,
                                &SourcePatchConfig {
                                    fast,
                                    min_peak,
                                    fingerprint_divisor,
                                    ..SourcePatchConfig::default()
                                },
                            );
                            if scenario == 4 {
                                assert!(result.is_empty());
                            }
                            if scenario == 0 && fingerprint_divisor == 2 {
                                assert_eq!(result.len(), 1);
                                assert_eq!(result[0].positions, [(13, 41), (65, 13)]);
                                let delta = 64.0f32 * (1.0f32 / 255.0);
                                assert_eq!(
                                    result[0].key.samples,
                                    vec![delta.to_bits() as i32; 25 * colors]
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// Keep the scalar predicate independent of the optimized integer shortcut.
fn scalar_same(a: [f32; 3], b: [f32; 3]) -> bool {
    a.into_iter()
        .zip(b)
        .all(|(a, b)| f64::from((a - b).abs()) <= 1e-4)
}

mod legacy {
    use super::scalar_same as same;
    use super::*;

    pub(super) fn legacy_background(
        image: DetectionView<'_>,
        config: &SourcePatchConfig,
        _work: &mut Work,
    ) -> Result<Option<Vec<usize>>> {
        let (pw, ph) = (image.width / SEED_SIDE, image.height / SEED_SIDE);
        if pw < 3 || ph < 3 {
            return Ok(None);
        }
        let mut seeds = zeros::<u8>(pw * ph)?;
        let mut count = 0;
        // RunOnPool(1, ph - 2) has an exclusive end in the frozen source. Keep its
        // asymmetric final seed row rather than silently widening the scan.
        for py in 1..ph - 2 {
            for px in 1..pw - 1 {
                let origin = py * SEED_SIDE * image.width + px * SEED_SIDE;
                let base = image.pick(origin);
                if !(0..SEED_SIDE).all(|y| {
                    (0..SEED_SIDE).all(|x| same(base, image.pick(origin + y * image.width + x)))
                }) {
                    continue;
                }
                let mut equal = 0;
                for y in py - 1..=py + 1 {
                    for x in px - 1..=px + 1 {
                        equal += usize::from(same(
                            base,
                            image.pick(y * SEED_SIDE * image.width + x * SEED_SIDE),
                        ));
                    }
                }
                if equal >= 8 {
                    seeds[py * pw + px] = 1;
                    count += 1;
                }
            }
        }
        if count == 0 {
            return Ok(None);
        }
        let pixels = image.width * image.height;
        let mut origins = zeros::<usize>(pixels)?;
        origins.fill(usize::MAX);
        let mut queue = VecDeque::new();
        queue
            .try_reserve((count * 16).min(65_536))
            .map_err(|_| Error::Limit("source patch background queue"))?;
        for py in 1..ph - 1 {
            for px in 1..pw - 1 {
                if seeds[py * pw + px] == 0 {
                    continue;
                }
                for y in py * SEED_SIDE..(py + 1) * SEED_SIDE {
                    for x in px * SEED_SIDE..(px + 1) * SEED_SIDE {
                        let position = y * image.width + x;
                        origins[position] = position;
                        if !config.fast {
                            queue.push_back((position, position));
                        }
                    }
                }
            }
        }
        if config.fast {
            for py in 1..ph - 1 {
                for px in 1..pw - 1 {
                    if seeds[py * pw + px] == 0
                        || (py - 1..=py + 1)
                            .all(|y| (px - 1..=px + 1).all(|x| seeds[y * pw + x] != 0))
                    {
                        continue;
                    }
                    for y in py * SEED_SIDE..(py + 1) * SEED_SIDE {
                        for x in px * SEED_SIDE..(px + 1) * SEED_SIDE {
                            let position = y * image.width + x;
                            let mut interior = true;
                            neighbors(image.width, image.height, position, |next| {
                                interior &= origins[next] != usize::MAX
                            });
                            if !interior {
                                queue.push_back((position, position));
                            }
                        }
                    }
                }
            }
        }
        #[cfg(test)]
        {
            _work.initial_queue = queue.len();
        }
        while let Some((position, source)) = queue.pop_front() {
            neighbors(image.width, image.height, position, |next| {
                if origins[next] != usize::MAX {
                    return;
                }
                let distance = (source % image.width).abs_diff(next % image.width)
                    + (source / image.width).abs_diff(next / image.width);
                if distance <= BACKGROUND_DISTANCE
                    && similar(image.pick(source), image.pick(next), 0.8)
                {
                    origins[next] = source;
                    queue.push_back((next, source));
                }
            });
        }
        Ok(Some(origins))
    }

    pub(super) fn legacy_detect_view(
        image: DetectionView<'_>,
        config: &SourcePatchConfig,
        work: &mut Work,
    ) -> Result<Vec<PatchInfo>> {
        let Some(background) = legacy_background(image, config, work)? else {
            return Ok(Vec::new());
        };
        let pixels = image.width * image.height;
        let limits = &config.limits;
        let mut visited = zeros::<u8>(pixels)?;
        let mut stack = Vec::new();
        let mut groups = BTreeMap::<Fingerprint, PatchInfo>::new();
        let mut positions = 0;
        for start in 0..pixels {
            if background[start] != usize::MAX || (config.fast && visited[start] != 0) {
                continue;
            }
            let (mut min_x, mut max_x) = (start % image.width, start % image.width);
            let (mut min_y, mut max_y) = (start / image.width, start / image.width);
            let mut border = None;
            let mut same_border = true;
            stack.push(start);
            if config.fast {
                visited[start] = 1;
            }
            #[cfg(test)]
            {
                work.component_pushes += 1;
            }
            while let Some(position) = stack.pop() {
                if !config.fast {
                    if visited[position] != 0 {
                        continue;
                    }
                    visited[position] = 1;
                }
                let (x, y) = (position % image.width, position / image.width);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                neighbors(image.width, image.height, position, |next| {
                    if background[next] == usize::MAX {
                        if config.fast {
                            if visited[next] != 0 {
                                return;
                            }
                            visited[next] = 1;
                        }
                        stack.push(next);
                        #[cfg(test)]
                        {
                            work.component_pushes += 1;
                        }
                    } else {
                        if config.fast
                            && (!same_border
                                || max_x - min_x >= limits.max_patch_size
                                || max_y - min_y >= limits.max_patch_size)
                        {
                            return;
                        }
                        if let Some(reference) = border {
                            same_border &=
                                similar(image.pick(background[next]), image.pick(reference), 0.03);
                        } else {
                            border = Some(background[next]);
                        }
                    }
                });
            }
            let Some(reference) = border else {
                continue;
            };
            if !same_border
                || max_x - min_x >= limits.max_patch_size
                || max_y - min_y >= limits.max_patch_size
                || positions >= limits.max_positions
            {
                continue;
            }
            let normalized_base = image.pick(reference);
            let has_similar = (min_y.saturating_sub(2)..(max_y + 3).min(image.height)).any(|y| {
                (min_x.saturating_sub(2)..(max_x + 3).min(image.width))
                    .any(|x| similar(normalized_base, image.pick(y * image.width + x), 0.03))
            });
            if !has_similar {
                continue;
            }
            let (width, height) = (max_x - min_x + 1, max_y - min_y + 1);
            let mut bytes = Vec::with_capacity(width * height * 3);
            let mut deltas = Vec::with_capacity(width * height * image.color_channels);
            let (mut too_small, mut too_big) = (true, false);
            for (c, &base) in normalized_base.iter().enumerate() {
                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        // Keep the source subtraction of normalized operands. The
                        // exact integer delta remains separate from its float key.
                        let position = y * image.width + x;
                        let normalized = image.pick(position)[c] - base;
                        // Source float-to-int quantization of NaN is undefined.
                        // Never create an arithmetic patch containing nonfinite
                        // deltas; those input words remain in the untouched plane.
                        if !normalized.is_finite() {
                            too_big = true;
                            continue;
                        }
                        let q = quantize(normalized, c);
                        too_small &= q > -config.min_peak && q < config.min_peak;
                        too_big |= !(i8::MIN as i32..=i8::MAX as i32).contains(&q);
                        bytes.push((q / config.fingerprint_divisor) as i8 as u8);
                        if image.color_channels == 3 || c == 0 {
                            deltas.push(image.patch_sample(position, reference, c, normalized));
                        }
                    }
                }
            }
            if too_small || too_big {
                #[cfg(test)]
                {
                    work.low_peak += usize::from(too_small);
                }
                continue;
            }
            let fingerprint = Fingerprint {
                width,
                height,
                bytes,
            };
            let position = (min_x as u32, min_y as u32);
            if let Some(existing) = groups.get_mut(&fingerprint) {
                // std::pair<QuantizedPatch, positions> sorts equal keys by (x,y),
                // not the detector's (y,x) scan order. Retain that representative.
                if position < existing.positions[0] {
                    existing.key.samples = deltas;
                    existing.positions.insert(0, position);
                } else {
                    existing.positions.push(position);
                }
                positions += 1;
            } else if groups.len() < limits.max_unique_patches {
                groups.insert(
                    fingerprint,
                    PatchInfo {
                        key: PatchKey {
                            width,
                            height,
                            samples: deltas,
                        },
                        positions: vec![position],
                    },
                );
                positions += 1;
            }
        }
        let mut info: Vec<_> = groups
            .into_values()
            .filter(|p| p.positions.len() >= limits.min_occurrences)
            .collect();
        if info
            .iter()
            .map(|p| p.key.width * p.key.height)
            .max()
            .unwrap_or(0)
            < limits.min_max_patch_area
        {
            return Ok(Vec::new());
        }
        for patch in &mut info {
            patch.positions.sort_unstable();
        }
        // Source area sort leaves equal-area order unspecified; use stable key
        // order for reproducible Rust output. This does not change pixel replay.
        info.sort_by_key(|p| std::cmp::Reverse(p.key.width * p.key.height));
        Ok(info)
    }
}
