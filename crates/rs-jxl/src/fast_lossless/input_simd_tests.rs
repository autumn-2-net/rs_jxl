// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.

use super::*;

const CANARY: i32 = -0x0123_4567;

fn view(bytes: &[u8], width: usize, channels: u8, depth: u8, big_endian: bool) -> ImageView<'_> {
    ImageView {
        pixels: bytes,
        width,
        height: 1,
        row_stride: width * usize::from(channels) * if depth <= 8 { 1 } else { 2 },
        channels,
        bit_depth: depth,
        big_endian,
    }
}

// Independent byte-arithmetic oracle: no production sample/loader/transform.
fn scalar(image: ImageView<'_>, x: usize, y: usize, width: usize) -> [[i32; 257]; 4] {
    let mut rows = [[CANARY; 257]; 4];
    let storage = if image.bit_depth <= 8 { 1 } else { 2 };
    for (c, row) in rows
        .iter_mut()
        .enumerate()
        .take(usize::from(image.channels))
    {
        for (i, value) in row[1..1 + width].iter_mut().enumerate() {
            let at = y * image.row_stride + ((x + i) * usize::from(image.channels) + c) * storage;
            let first = i32::from(image.pixels[at]);
            *value = if storage == 1 {
                first
            } else {
                let second = i32::from(image.pixels[at + 1]);
                if image.big_endian {
                    256 * first + second
                } else {
                    first + 256 * second
                }
            };
        }
    }
    rows
}

fn check(image: ImageView<'_>, x: usize, y: usize, width: usize) {
    assert!(std::is_x86_feature_detected!("avx2"));
    let expected = scalar(image, x, y, width);
    let mut actual = [[CANARY; 257]; 4];
    // SAFETY: explicit AVX2 detection above; the entry checks all memory bounds.
    unsafe { load_row(image, x, y, width, &mut actual) }.unwrap();
    assert_eq!(
        actual, expected,
        "channels={} depth={} big_endian={} x={x} y={y} width={width}",
        image.channels, image.bit_depth, image.big_endian
    );
}

fn scalar_transformed(image: ImageView<'_>, x: usize, y: usize, width: usize) -> [[i32; 257]; 4] {
    let mut rows = scalar(image, x, y, width);
    if image.channels >= 3 {
        let [red, green, blue, _] = &mut rows;
        for ((red, green), blue) in red[1..1 + width]
            .iter_mut()
            .zip(&mut green[1..1 + width])
            .zip(&mut blue[1..1 + width])
        {
            let (r, g, b) = (*red, *green, *blue);
            // Independent nonnegative-average formulation. For valid input,
            // floor((r+b)/2) == b + arithmetic_shift_right(r-b, 1).
            let average = (r + b) / 2;
            *red = (g + average) / 2;
            *green = r - b;
            *blue = g - average;
        }
    }
    rows
}

fn check_transformed(image: ImageView<'_>, x: usize, y: usize, width: usize) {
    assert!(std::is_x86_feature_detected!("avx2"));
    let expected = scalar_transformed(image, x, y, width);
    let mut actual = [[CANARY; 257]; 4];
    // SAFETY: detected AVX2; the helper checks address bounds, and these
    // fixtures supply samples within the declared depth as region validation does.
    unsafe { load_transformed_row(image, x, y, width, &mut actual) }.unwrap();
    assert_eq!(
        actual, expected,
        "transformed channels={} depth={} big_endian={} x={x} y={y} width={width}",
        image.channels, image.bit_depth, image.big_endian
    );
}

fn fill_samples(
    bytes: &mut [u8],
    channels: u8,
    depth: u8,
    big_endian: bool,
    value: impl Fn(usize, usize) -> u16,
) {
    let storage = if depth <= 8 { 1 } else { 2 };
    let maximum = (1u32 << depth) - 1;
    for (index, sample) in bytes.chunks_exact_mut(storage).enumerate() {
        let word = value(index / usize::from(channels), index % usize::from(channels));
        assert!(u32::from(word) <= maximum);
        if storage == 1 {
            sample[0] = word as u8;
        } else {
            sample.copy_from_slice(&if big_endian {
                word.to_be_bytes()
            } else {
                word.to_le_bytes()
            });
        }
    }
}

#[test]
fn transformed_rows_exhaust_every_valid_sample_range_per_channel() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for depth in 1u8..=16 {
        let storage = if depth <= 8 { 1 } else { 2 };
        let range = 1usize << depth;
        for channels in 1u8..=4 {
            for big_endian in [false, true] {
                for base in (0..range).step_by(256) {
                    let offset = 1 + base / 256 % 31;
                    let length = 256 * usize::from(channels) * storage;
                    let mut bytes = vec![0xa7; offset + length + 32];
                    let input = &mut bytes[offset..offset + length];
                    fill_samples(input, channels, depth, big_endian, |x, c| {
                        // Odd multipliers permute the full power-of-two range
                        // independently in each channel, including high alpha.
                        (((base + x) * (2 * c + 1) + c * 7919) & (range - 1)) as u16
                    });
                    check_transformed(view(input, 256, channels, depth, big_endian), 0, 0, 256);
                    assert!(bytes[..offset].iter().all(|&b| b == 0xa7));
                    assert!(bytes[offset + length..].iter().all(|&b| b == 0xa7));
                }
            }
        }
    }
}

#[test]
fn transformed_rows_all_widths_alignments_and_depth_profiles_preserve_guards() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for depth in [1, 8, 9, 14, 15, 16] {
        let storage = if depth <= 8 { 1 } else { 2 };
        let maximum = (1usize << depth) - 1;
        for channels in 1u8..=4 {
            let pixel_size = usize::from(channels) * storage;
            for big_endian in [false, true] {
                let mut samples = vec![0; 256 * pixel_size];
                fill_samples(&mut samples, channels, depth, big_endian, |x, c| {
                    ((x * 8191 + c * 7177 + 37) & maximum) as u16
                });
                for offset in 0..32 {
                    let mut bytes = vec![0xa7; offset + samples.len() + 32];
                    bytes[offset..offset + samples.len()].copy_from_slice(&samples);
                    for width in 0..=256 {
                        let input = &bytes[offset..offset + width * pixel_size];
                        check_transformed(
                            view(input, width, channels, depth, big_endian),
                            0,
                            0,
                            width,
                        );
                    }
                    assert_eq!(&bytes[offset..offset + samples.len()], samples.as_slice());
                    assert!(bytes[..offset].iter().all(|&b| b == 0xa7));
                    assert!(bytes[offset + samples.len()..].iter().all(|&b| b == 0xa7));
                }
            }
        }
    }
}

#[test]
fn transformed_extreme_colors_and_negative_odd_chroma_match_scalar_tails() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for depth in 1u8..=16 {
        let maximum = ((1u32 << depth) - 1) as u16;
        let storage = if depth <= 8 { 1 } else { 2 };
        let colors = [
            [0, 0, 0],
            [0, 0, maximum],
            [0, maximum, 0],
            [0, maximum, maximum],
            [maximum, 0, 0],
            [maximum, 0, maximum],
            [maximum, maximum, 0],
            [maximum, maximum, maximum],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [maximum - 1, 0, maximum],
            [maximum, 1, 0],
            [1, maximum, maximum - 1],
            [maximum / 2, 0, maximum / 2 + 1],
            [maximum / 2 + 1, maximum, maximum / 2],
        ];
        for channels in [3u8, 4] {
            for big_endian in [false, true] {
                let mut bytes = vec![0; 256 * usize::from(channels) * storage];
                fill_samples(&mut bytes, channels, depth, big_endian, |x, c| {
                    if c == 3 {
                        if x.is_multiple_of(2) { 0 } else { maximum }
                    } else {
                        colors[x % colors.len()][c]
                    }
                });
                let image = view(&bytes, 256, channels, depth, big_endian);
                for x in 0..16 {
                    // Every corner occupies both a vector lane and a scalar
                    // tail position; zero alpha does not suppress hidden RGB.
                    for width in [1, 15, 16, 17, 31, 32, 33, 239] {
                        check_transformed(image, x, 0, width);
                    }
                }
            }
        }
    }
}

#[test]
fn signed_chroma_widening_preserves_every_i16_word_and_lane_order() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for base in (0..=u16::MAX as u32).step_by(16) {
        let input: [u16; 16] = std::array::from_fn(|lane| (base + lane as u32) as u16);
        let mut output = [CANARY; 18];
        // SAFETY: detected AVX2, exactly sixteen readable input words and
        // sixteen writable output values between initialized guard cells.
        unsafe {
            let value = _mm256_loadu_si256(input.as_ptr().cast());
            store_signed_widened(value, output.as_mut_ptr().add(1));
        }
        assert_eq!(output[0], CANARY);
        assert_eq!(output[17], CANARY);
        for (&actual, &word) in output[1..17].iter().zip(&input) {
            let expected = if word & 0x8000 == 0 {
                i32::from(word)
            } else {
                i32::from(word) - 65536
            };
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn every_u8_u16_value_in_each_channel_and_both_byte_orders() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for depth in [8, 16] {
        let storage = if depth == 8 { 1 } else { 2 };
        let range = 1usize << depth;
        for channels in 1u8..=4 {
            for big_endian in [false, true] {
                for base in (0..range).step_by(256) {
                    let misalignment = 1 + base / 256 % 31;
                    let length = 256 * usize::from(channels) * storage;
                    let mut bytes = vec![0xa7; misalignment + length + 32];
                    let input = &mut bytes[misalignment..misalignment + length];
                    for x in 0..256 {
                        for c in 0..usize::from(channels) {
                            // For each channel this is a permutation of the
                            // complete word range, not just repeated gray data.
                            let value = ((base + x + c * 7919) % range) as u16;
                            let at = (x * usize::from(channels) + c) * storage;
                            if storage == 1 {
                                input[at] = value as u8;
                            } else {
                                input[at..at + 2].copy_from_slice(&if big_endian {
                                    value.to_be_bytes()
                                } else {
                                    value.to_le_bytes()
                                });
                            }
                        }
                    }
                    check(view(input, 256, channels, depth, big_endian), 0, 0, 256);
                    assert!(bytes[..misalignment].iter().all(|&b| b == 0xa7));
                    assert!(bytes[misalignment + length..].iter().all(|&b| b == 0xa7));
                }
            }
        }
    }
}

#[test]
fn all_widths_alignments_channel_layouts_and_tails_preserve_output_guards() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut bytes = [0u8; 32 + 256 * 8 + 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = ((i * 137 + i / 3 * 19 + 71) & 255) as u8;
    }
    for depth in [8, 16] {
        let storage = if depth == 8 { 1 } else { 2 };
        for channels in 1u8..=4 {
            for big_endian in [false, true] {
                for width in 0..=256 {
                    let length = width * usize::from(channels) * storage;
                    for offset in 0..32 {
                        let input = &bytes[offset..offset + length];
                        check(view(input, width, channels, depth, big_endian), 0, 0, width);
                    }
                }
            }
        }
    }
}

#[test]
fn strided_subrows_use_absolute_coordinates_without_final_row_padding() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for depth in 1u8..=16 {
        let storage = if depth <= 8 { 1 } else { 2 };
        for channels in 1u8..=4 {
            for big_endian in [false, true] {
                let width = 277;
                let height = 3;
                let row = width * usize::from(channels) * storage;
                let stride = row + 7;
                let length = (height - 1) * stride + row;
                let mut bytes = vec![0xa5; 3 + length];
                let input = &mut bytes[3..];
                let max = (1u32 << depth) - 1;
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..usize::from(channels) {
                            let word = ((x as u32 * 8191 + y as u32 * 11939 + c as u32 * 7177)
                                & max) as u16;
                            let at = y * stride + (x * usize::from(channels) + c) * storage;
                            if storage == 1 {
                                input[at] = word as u8;
                            } else {
                                input[at..at + 2].copy_from_slice(&if big_endian {
                                    word.to_be_bytes()
                                } else {
                                    word.to_le_bytes()
                                });
                            }
                        }
                    }
                }
                let image = ImageView {
                    height,
                    row_stride: stride,
                    ..view(input, width, channels, depth, big_endian)
                };
                for y in 0..height {
                    for (x, count) in [
                        (0, 0),
                        (0, 1),
                        (1, 15),
                        (3, 16),
                        (5, 17),
                        (7, 255),
                        (21, 256),
                        (276, 1),
                        (277, 0),
                    ] {
                        check(image, x, y, count);
                        check_transformed(image, x, y, count);
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_geometry_and_buffers_fail_without_touching_destination() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let bytes = [0; 2 * 33 * 8];
    let valid = ImageView {
        height: 2,
        ..view(&bytes, 33, 4, 16, false)
    };
    let mut cases = vec![
        (valid, 0, 0, 257),
        (valid, 1, 0, 33),
        (valid, usize::MAX, 0, 16),
        (valid, 0, 2, 16),
        (valid, 0, usize::MAX, 16),
        (
            ImageView {
                row_stride: 33 * 8 - 1,
                ..valid
            },
            0,
            0,
            16,
        ),
        (
            ImageView {
                pixels: &bytes[..16 * 8 - 1],
                ..valid
            },
            0,
            0,
            16,
        ),
        (
            ImageView {
                pixels: &bytes[..bytes.len() - 1],
                ..valid
            },
            17,
            1,
            16,
        ),
        (
            ImageView {
                width: usize::MAX,
                row_stride: usize::MAX,
                ..valid
            },
            0,
            0,
            1,
        ),
        (
            ImageView {
                height: 3,
                row_stride: usize::MAX,
                ..valid
            },
            0,
            2,
            1,
        ),
        (
            ImageView {
                row_stride: usize::MAX,
                ..valid
            },
            1,
            1,
            1,
        ),
        (
            ImageView {
                row_stride: usize::MAX - 4,
                ..valid
            },
            0,
            1,
            1,
        ),
    ];
    for channels in [0, 5, u8::MAX] {
        cases.push((ImageView { channels, ..valid }, 0, 0, 16));
    }
    for bit_depth in [0, 17, u8::MAX] {
        cases.push((ImageView { bit_depth, ..valid }, 0, 0, 16));
    }
    for (image, x, y, width) in cases {
        let mut rows = [[CANARY; 257]; 4];
        // SAFETY: AVX2 is detected; intentionally malformed descriptors must
        // return an error because bounds, not caller promises, protect loads.
        assert!(unsafe { load_row(image, x, y, width, &mut rows) }.is_err());
        assert_eq!(rows, [[CANARY; 257]; 4]);
        // SAFETY: the same CPU-only precondition applies to the fused entry.
        assert!(unsafe { load_transformed_row(image, x, y, width, &mut rows) }.is_err());
        assert_eq!(rows, [[CANARY; 257]; 4]);
    }
}

#[test]
fn endian_swap_and_unsigned_upcast_preserve_every_word() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for base in (0..=u16::MAX as u32).step_by(16) {
        let input: [u16; 16] = std::array::from_fn(|lane| (base + lane as u32) as u16);
        let mut output = [CANARY; 18];
        // SAFETY: detected AVX2; input has 32 readable bytes, and the store
        // has exactly sixteen writable i32 cells between the two canaries.
        unsafe {
            let value = _mm256_loadu_si256(input.as_ptr().cast());
            store_widened(swap_endian(value), output.as_mut_ptr().add(1));
        }
        assert_eq!(output[0], CANARY);
        assert_eq!(output[17], CANARY);
        for (actual, &word) in output[1..17].iter().zip(&input) {
            let expected = u32::from(word & 255) * 256 + u32::from(word >> 8);
            assert_eq!(*actual, expected as i32);
        }
    }
}

#[test]
fn preserves_raw_words_instead_of_clamping_to_declared_depth() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let bytes = [255; 33 * 8];
    for depth in 1..=16 {
        for channels in 1..=4 {
            for big_endian in [false, true] {
                check(view(&bytes, 33, channels, depth, big_endian), 0, 0, 33);
            }
        }
    }
}

// Test-only OS memory protection, not native codec linkage. The inaccessible
// pages detect actual overreads that input canaries alone cannot detect.
#[cfg(windows)]
mod protected {
    use super::*;
    use std::ffi::c_void;

    // A 64 KiB boundary is page-aligned on Windows x86/x86_64. Keep a readable
    // block between inaccessible blocks; no target other than x86 is compiled.
    const BLOCK: usize = 65536;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(address: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualProtect(address: *mut c_void, size: usize, protect: u32, old: *mut u32) -> i32;
        fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32;
    }

    struct Guarded(*mut u8);
    impl Guarded {
        fn new() -> Self {
            // SAFETY: reserve/commit three blocks as PAGE_NOACCESS, then make
            // only the middle block PAGE_READWRITE. No references exist yet.
            let allocation = unsafe { VirtualAlloc(std::ptr::null_mut(), 3 * BLOCK, 0x3000, 0x01) };
            assert!(!allocation.is_null(), "VirtualAlloc failed");
            let guarded = Self(allocation.cast());
            let mut old = 0;
            // SAFETY: the middle block belongs to the live allocation and
            // is page-aligned; old points to a writable protection word.
            let ok = unsafe { VirtualProtect(guarded.0.add(BLOCK).cast(), BLOCK, 0x04, &mut old) };
            assert_ne!(ok, 0, "VirtualProtect failed");
            guarded
        }

        fn input(&mut self, length: usize, at_end: bool) -> &mut [u8] {
            assert!(length <= BLOCK);
            let offset = BLOCK + if at_end { BLOCK - length } else { 0 };
            // SAFETY: this mutable borrow exclusively covers initialized,
            // readable/writable bytes in the middle committed block only.
            unsafe { std::slice::from_raw_parts_mut(self.0.add(offset), length) }
        }
    }
    impl Drop for Guarded {
        fn drop(&mut self) {
            // SAFETY: this is the original allocation and no slices outlive it.
            unsafe {
                VirtualFree(self.0.cast(), 0, 0x8000);
            }
        }
    }

    #[test]
    fn transformed_rows_touch_protected_input_and_output_boundaries() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut source = Guarded::new();
        let mut destination = Guarded::new();
        for depth in [1, 8, 9, 14, 15, 16] {
            let storage = if depth <= 8 { 1 } else { 2 };
            let maximum = (1usize << depth) - 1;
            for channels in 1u8..=4 {
                for big_endian in [false, true] {
                    for width in 0..=256 {
                        for input_at_end in [false, true] {
                            let input =
                                source.input(width * usize::from(channels) * storage, input_at_end);
                            fill_samples(input, channels, depth, big_endian, |x, c| {
                                ((x * 8191 + c * 7177) & maximum) as u16
                            });
                            let image = view(input, width, channels, depth, big_endian);
                            let expected = scalar_transformed(image, 0, 0, width);
                            for output_at_end in [false, true] {
                                let output = destination
                                    .input(std::mem::size_of::<[[i32; 257]; 4]>(), output_at_end);
                                let pointer = output.as_mut_ptr().cast::<[[i32; 257]; 4]>();
                                assert_eq!(pointer as usize % std::mem::align_of::<i32>(), 0);
                                // SAFETY: aligned initialized memory of exactly
                                // the array size, exclusively borrowed here.
                                let rows = unsafe { &mut *pointer };
                                *rows = [[CANARY; 257]; 4];
                                // SAFETY: detected AVX2; exact buffers, valid depth.
                                unsafe { load_transformed_row(image, 0, 0, width, rows) }.unwrap();
                                assert_eq!(*rows, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exact_input_segments_touch_both_protected_boundaries_without_overreads() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut guarded = Guarded::new();
        for depth in [8, 16] {
            let storage = if depth == 8 { 1 } else { 2 };
            for channels in 1u8..=4 {
                for big_endian in [false, true] {
                    for width in 0..=256 {
                        for at_end in [false, true] {
                            let input =
                                guarded.input(width * usize::from(channels) * storage, at_end);
                            for (i, byte) in input.iter_mut().enumerate() {
                                *byte = ((i * 131 + 37) & 255) as u8;
                            }
                            check(view(input, width, channels, depth, big_endian), 0, 0, width);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn destination_rows_touch_protected_boundaries_without_overwrites() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut guarded = Guarded::new();
        let input: [u8; 256 * 8] = std::array::from_fn(|i| ((i * 73 + 91) & 255) as u8);
        for depth in [8, 16] {
            for channels in 1..=4 {
                for big_endian in [false, true] {
                    for width in 0..=256 {
                        let image = view(&input, width, channels, depth, big_endian);
                        let expected = scalar(image, 0, 0, width);
                        for at_end in [false, true] {
                            let output =
                                guarded.input(std::mem::size_of::<[[i32; 257]; 4]>(), at_end);
                            let pointer = output.as_mut_ptr().cast::<[[i32; 257]; 4]>();
                            assert_eq!(pointer as usize % std::mem::align_of::<i32>(), 0);
                            // SAFETY: the page-aligned allocation and array size
                            // give aligned, initialized storage of exactly this
                            // type. output is exclusively borrowed until rows'
                            // last use, and no typed reference escapes the test.
                            let rows = unsafe { &mut *pointer };
                            *rows = [[CANARY; 257]; 4];
                            // SAFETY: AVX2 is detected; input bounds are checked.
                            unsafe { load_row(image, 0, 0, width, rows) }.unwrap();
                            assert_eq!(*rows, expected);
                        }
                    }
                }
            }
        }
    }
}
