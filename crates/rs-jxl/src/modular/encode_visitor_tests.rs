//! Traversal equality only; these tests do not establish an ISA speedup.

use super::*;

#[derive(Clone, Copy, Debug)]
enum Kernel {
    Portable,
    Dispatched,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

#[cfg(target_arch = "x86_64")]
fn avx512_supported() -> bool {
    std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512vl")
}

fn kernels() -> Vec<Kernel> {
    let kernels = vec![Kernel::Portable, Kernel::Dispatched];
    #[cfg(target_arch = "x86_64")]
    {
        let mut kernels = kernels;
        if std::is_x86_feature_detected!("avx2") {
            kernels.push(Kernel::Avx2);
        }
        if avx512_supported() {
            kernels.push(Kernel::Avx512);
        }
        kernels
    }
    #[cfg(not(target_arch = "x86_64"))]
    kernels
}

#[derive(Debug, PartialEq, Eq)]
struct Pixel {
    channel: usize,
    position: usize,
    neighbors: Neighbors,
    prediction: i32,
    properties: Vec<i32>,
    value: i32,
}

#[allow(clippy::too_many_arguments)]
fn trace(
    kernel: Kernel,
    planes: &[Channel],
    channel: usize,
    wp: &weighted::Header,
    use_wp: bool,
    previous_channels: usize,
    fail_at: Option<usize>,
) -> (Vec<Pixel>, Result<()>) {
    let mut pixels = Vec::new();
    let visitor = |channel, position, neighbors, prediction, properties: &[i32], value| {
        pixels.push(Pixel {
            channel,
            position,
            neighbors,
            prediction,
            properties: properties.to_vec(),
            value,
        });
        if fail_at == Some(position) {
            Err(Error::Limit("visitor sentinel"))
        } else {
            Ok(())
        }
    };
    let result = match kernel {
        Kernel::Portable => {
            visit_channel_impl(planes, channel, wp, use_wp, previous_channels, 37, visitor)
        }
        Kernel::Dispatched => {
            visit_channel(planes, channel, wp, use_wp, previous_channels, 37, visitor)
        }
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => {
            assert!(std::is_x86_feature_detected!("avx2"));
            // SAFETY: this direct test entry is guarded by the AVX2 CPU check.
            unsafe {
                visit_channel_avx2(planes, channel, wp, use_wp, previous_channels, 37, visitor)
            }
        }
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 => {
            assert!(avx512_supported());
            // SAFETY: this direct test entry checks AVX512F/BW/DQ/VL above.
            unsafe {
                visit_channel_avx512(planes, channel, wp, use_wp, previous_channels, 37, visitor)
            }
        }
    };
    (pixels, result)
}

fn channel(width: usize, height: usize, seed: u32) -> Channel {
    let mut random = seed;
    let samples = (0..width * height)
        .map(|i| {
            random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match i % 11 {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => -1,
                3 => 0,
                4 => 1,
                _ => random as i32,
            }
        })
        .collect();
    Channel {
        width,
        height,
        hshift: 0,
        vshift: 0,
        samples,
    }
}

fn planes(width: usize, height: usize) -> Vec<Channel> {
    let mut shifted = channel(width, height, 7);
    shifted.hshift = 1;
    vec![
        channel(width, height, 1),
        shifted,
        channel(width, height, 3),
        channel(width + 1, height, 5),
        channel(width, height, 9),
    ]
}

#[test]
fn channel_dispatch_and_supported_isas_match_portable_all_weighted_presets() {
    let kernels = kernels();
    for width in [1, 2, 8, 9, 17, 257] {
        for height in [1, 2, 5] {
            let planes = planes(width, height);
            for preset in 0..5 {
                let wp = weighted::Header::preset(preset).unwrap();
                for use_wp in [false, true] {
                    for previous_channels in [0, 1, 2, 4] {
                        for c in [0, 4] {
                            let (expected, result) = trace(
                                Kernel::Portable,
                                &planes,
                                c,
                                &wp,
                                use_wp,
                                previous_channels,
                                None,
                            );
                            result.unwrap();
                            assert_eq!(expected.len(), width * height);
                            for (position, pixel) in expected.iter().enumerate() {
                                assert_eq!(pixel.position, position);
                                assert_eq!(pixel.channel, c);
                                assert_eq!(pixel.value, planes[c].samples[position]);
                                assert_eq!(pixel.properties.len(), 16 + 4 * previous_channels);
                                if !use_wp {
                                    assert_eq!(pixel.prediction, 0);
                                    assert_eq!(pixel.properties[15], 0);
                                }
                            }
                            for &kernel in &kernels[1..] {
                                let (actual, result) =
                                    trace(kernel, &planes, c, &wp, use_wp, previous_channels, None);
                                result.unwrap();
                                assert_eq!(
                                    actual, expected,
                                    "{kernel:?}, width={width}, height={height}, preset={preset}, use_wp={use_wp}, previous_channels={previous_channels}, c={c}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn channel_dispatch_and_supported_isas_preserve_flat_extreme_channels() {
    let kernels = kernels();
    for width in [1, 2, 8, 9, 17, 257] {
        for value in [i32::MIN, i32::MAX, 0] {
            let mut plane = channel(width, 5, 1);
            plane.samples.fill(value);
            let planes = [plane];
            for preset in 0..5 {
                let wp = weighted::Header::preset(preset).unwrap();
                let (expected, result) = trace(Kernel::Portable, &planes, 0, &wp, true, 1, None);
                result.unwrap();
                for &kernel in &kernels[1..] {
                    let (actual, result) = trace(kernel, &planes, 0, &wp, true, 1, None);
                    result.unwrap();
                    assert_eq!(
                        actual, expected,
                        "{kernel:?}, width={width}, value={value}, preset={preset}"
                    );
                }
            }
        }
    }
}

#[test]
fn channel_callback_error_preserves_exact_prefix() {
    let kernels = kernels();
    for width in [1, 2, 8, 9, 17, 257] {
        let planes = planes(width, 5);
        for preset in 0..5 {
            let wp = weighted::Header::preset(preset).unwrap();
            for use_wp in [false, true] {
                let (expected, result) = trace(Kernel::Portable, &planes, 4, &wp, use_wp, 4, None);
                result.unwrap();
                for fail_at in [0, width - 1, 2 * width + width / 2, width * 5 - 1] {
                    for &kernel in &kernels {
                        let (actual, result) =
                            trace(kernel, &planes, 4, &wp, use_wp, 4, Some(fail_at));
                        assert!(matches!(result, Err(Error::Limit("visitor sentinel"))));
                        assert_eq!(
                            actual,
                            expected[..=fail_at],
                            "{kernel:?}, width={width}, preset={preset}, use_wp={use_wp}, fail_at={fail_at}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn empty_channel_never_calls_the_visitor() {
    for kernel in kernels() {
        for (width, height) in [(0, 5), (17, 0)] {
            let planes = [channel(width, height, 1)];
            for use_wp in [false, true] {
                let (pixels, result) = trace(
                    kernel,
                    &planes,
                    0,
                    &weighted::Header::default(),
                    use_wp,
                    2,
                    Some(0),
                );
                result.unwrap();
                assert!(pixels.is_empty());
            }
        }
    }
}
