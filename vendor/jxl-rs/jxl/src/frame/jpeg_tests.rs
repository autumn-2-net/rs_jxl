// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::*;
use crate::bit_reader::BitReader;
use crate::headers::encodings::UnconditionalCoder;
use crate::headers::frame_header::FrameHeaderNonserialized;
use crate::image::{Image, Rect};

fn header(width: u32, height: u32, ycbcr: bool, modes: [u32; 3]) -> FrameHeader {
    // The encoder's one-pass JPEG frame header, without the TOC.
    let mut bits = Vec::new();
    let mut put = |n: usize, value: u32| bits.extend((0..n).map(|i| ((value >> i) & 1) as u8));
    for (n, value) in [
        (1, 0),
        (2, 0),
        (1, 0),
        (2, 2),
        (8, 111),
        (1, u32::from(ycbcr)),
    ] {
        put(n, value);
    }
    if ycbcr {
        for mode in modes {
            put(2, mode);
        }
    }
    for (n, value) in [
        (2, 0),
        (2, 0),
        (1, 0),
        (2, 0),
        (1, 1),
        (2, 0),
        (1, 0),
        (1, 0),
        (2, 0),
        (2, 0),
        (2, 0),
    ] {
        put(n, value);
    }
    let bytes: Vec<u8> = bits
        .chunks(8)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0, |byte, (i, bit)| byte | (*bit << i))
        })
        .collect();
    let nonserialized = FrameHeaderNonserialized {
        xyb_encoded: false,
        num_extra_channels: 0,
        extra_channel_info: vec![],
        have_animation: false,
        have_timecode: false,
        img_width: width,
        img_height: height,
    };
    let mut header =
        FrameHeader::read_unconditional(&(), &mut BitReader::new(&bytes), &nonserialized).unwrap();
    header.postprocess(&nonserialized);
    header
}

fn tables() -> QuantEncoding {
    QuantEncoding::Raw {
        qtable: (0..192)
            .map(|i| {
                let c = i / 64;
                let k = i % 64;
                (3 * c + (k % 8) * 8 + k / 8 + 1) as i32
            })
            .collect(),
        qtable_den: crate::util::f16::from_bits(0x1004).to_f32(),
    }
}

fn dc(c: usize, x: usize, y: usize) -> i32 {
    c as i32 * 100 + (x % 31) as i32 - (y % 17) as i32 - 200
}
fn ac(c: usize, x: usize, y: usize, k: usize) -> i32 {
    (c * 70 + k + x % 13) as i32 - (y % 11) as i32 - 100
}

fn restore_ac(residual: i32, luma: i32, q_y: u16, q_c: u16, factor: i8) -> Result<i16> {
    let ratio = FIXED_ONE * i32::from(q_y) / i32::from(q_c);
    let mut cache = CflScaleCache::new([ratio; 64]);
    let scaled =
        cache.for_factor(factor)[1].ok_or(unsupported("JPEG capture CfL scale overflow"))?;
    restore_ac_scaled(residual, luma, scaled)
}

fn reference_restore_ac(
    residual: i32,
    luma: i32,
    q_y: u16,
    q_c: u16,
    factor: i8,
) -> std::result::Result<i16, &'static str> {
    // The original operation order, with independent wide arithmetic and an
    // explicit i32 check at each intermediate, not just on the final value.
    let ratio = 2048 * i64::from(q_y) / i64::from(q_c);
    let scale = 2048 * i64::from(factor) / 84;
    let product = i32::try_from(ratio * scale).map_err(|_| "JPEG capture CfL scale overflow")?;
    let scaled = i32::try_from(i64::from(product) + 1024)
        .map_err(|_| "JPEG capture CfL scale overflow")?
        >> 11;
    let product = i32::try_from(i64::from(luma) * i64::from(scaled))
        .map_err(|_| "JPEG capture CfL prediction overflow")?;
    let prediction = i32::try_from(i64::from(product) + 1024)
        .map_err(|_| "JPEG capture CfL prediction overflow")?
        >> 11;
    let value = i32::try_from(i64::from(residual) + i64::from(prediction))
        .map_err(|_| "JPEG capture AC coefficient overflow")?;
    if value < -4095 || value > 4095 {
        return Err("JPEG capture AC coefficient is outside the 8-bit JPEG range");
    }
    Ok(value as i16)
}

fn ac_error_reason(error: Error) -> &'static str {
    match error {
        Error::ApiUsageError(reason) => reason,
        other => panic!("Unexpected coefficient error: {other}"),
    }
}

#[test]
fn cached_cfl_matches_checked_reference_for_all_factors_and_boundaries() {
    let quantization = [
        (1, 1),
        (3, 7),
        (7, 3),
        (65535, 65535),
        (65535, 1),
        (1, 65535),
        (84, 83),
        (1023, 2),
        (1024, 2),
        (1025, 2),
        (65535, 84),
        (65535, 83),
        (32768, 127),
        (1, 23),
    ];
    let ratios = std::array::from_fn(|k| {
        let (y, c) = quantization[k % quantization.len()];
        2048 * i32::from(y) / i32::from(c)
    });
    let mut cache = CflScaleCache::new(ratios);
    for factor in (i8::MIN..=i8::MAX).chain((i8::MIN..=i8::MAX).rev()) {
        // Repeated factors exercise reuse, not just cache invalidation.
        for _ in 0..2 {
            let scales = cache.for_factor(factor);
            for (k, &scale) in scales.iter().enumerate().skip(1) {
                let (q_y, q_c) = quantization[k % quantization.len()];
                for (residual, luma) in [
                    (0, 0),
                    (-4095, -4095),
                    (4095, 4095),
                    (i32::MIN, 0),
                    (i32::MAX, 0),
                    (0, i32::MIN),
                    (0, i32::MAX),
                    (0, i32::MAX - 1023),
                    (0, i32::MIN + 1023),
                    (-4096, 2047),
                    (4096, -2047),
                    (-1, -1),
                    (0, -1),
                    (0, 1),
                    (17, -199),
                    (-17, 199),
                    (i32::MAX, 1),
                    (i32::MIN, -1),
                ] {
                    let actual = scale
                        .ok_or(unsupported("JPEG capture CfL scale overflow"))
                        .and_then(|s| restore_ac_scaled(residual, luma, s))
                        .map_err(ac_error_reason);
                    assert_eq!(
                        actual,
                        reference_restore_ac(residual, luma, q_y, q_c, factor),
                        "factor={factor}, q_y={q_y}, q_c={q_c}, residual={residual}, luma={luma}",
                    );
                }
            }
        }
    }
}

#[test]
fn zero_factor_capture_preserves_range_checks_and_skips_dc() {
    let header = header(8, 8, true, [0; 3]);
    let capture = JpegCapture::new(&header, false).unwrap();
    capture
        .set_quantization(&QuantEncoding::Raw {
            qtable: (0..192)
                .map(|i| if i / 64 == 1 { 65535 } else { 1 })
                .collect(),
            qtable_den: 1.0 / 2040.0,
        })
        .unwrap();
    let mut group = capture.new_group(&header, 0).unwrap();
    let raw: [[i32; 64]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|k| {
            if k == 0 {
                i32::MIN
            } else {
                [-4095, 4095, 0][(k + c) % 3]
            }
        })
    });
    group
        .capture_block(&header, 0, 0, 0, raw.each_ref(), [0, 127, 0])
        .unwrap();
    for c in 0..3 {
        assert_eq!(group.coefficients[c][0], 0);
        for k in 1..64 {
            assert_eq!(
                i32::from(group.coefficients[c][k]),
                raw[c][(k % 8) * 8 + k / 8]
            );
        }
    }
    for c in 0..3 {
        for invalid in [i32::MIN, -4096, 4096, i32::MAX] {
            let mut invalid_raw = raw;
            invalid_raw[c][8] = invalid;
            let error = group
                .capture_block(&header, 0, 0, 0, invalid_raw.each_ref(), [0; 3])
                .unwrap_err();
            assert_eq!(
                ac_error_reason(error),
                "JPEG capture AC coefficient is outside the 8-bit JPEG range"
            );
        }
    }
}

#[test]
fn group_cfl_cache_tracks_factor_changes_and_quantization_transposition() {
    let header = header(137, 73, true, [0; 3]);
    let quantization: [[u16; 64]; 3] =
        std::array::from_fn(|c| std::array::from_fn(|k| ((k * 3 + c * 7) % 23 + 1) as u16));
    let capture = JpegCapture::new(&header, false).unwrap();
    capture
        .set_quantization(&QuantEncoding::Raw {
            qtable: (0..192)
                .map(|i| i32::from(quantization[i / 64][(i % 8) * 8 + (i % 64) / 8]))
                .collect(),
            qtable_den: 1.0 / 2040.0,
        })
        .unwrap();
    let mut data = capture.new_group(&header, 0).unwrap();
    let rect = header.block_group_rect(0);
    let factors = [-128, 127, 0, -42, 84, 0, 1, -1];
    for by in 0..rect.size.1 {
        for bx in 0..rect.size.0 {
            let index = by * rect.size.0 + bx;
            let factors = [
                factors[(index / 3) % factors.len()],
                127,
                factors[index % factors.len()],
            ];
            let raw = std::array::from_fn(|c| {
                std::array::from_fn(|k| ((k + c * 3 + index * 7) % 31) as i32 - 15)
            });
            data.capture_block(&header, 0, bx, by, raw.each_ref(), factors)
                .unwrap();
            for c in 0..3 {
                for k in 1..64 {
                    let transposed = (k % 8) * 8 + k / 8;
                    let expected = reference_restore_ac(
                        raw[c][transposed],
                        raw[1][transposed],
                        quantization[1][k],
                        quantization[c][k],
                        if c == 1 { 0 } else { factors[c] },
                    )
                    .unwrap();
                    assert_eq!(data.coefficients[c][index * 64 + k], expected);
                }
            }
        }
    }
}

#[test]
fn cfl_cache_preserves_error_order_and_valid_prefix_writes() {
    let header = header(8, 8, true, [0; 3]);
    let capture = JpegCapture::new(&header, false).unwrap();
    let mut quantization = [1; 192];
    quantization[64] = 65535; // DC is unused even when its CfL scale overflows.
    quantization[64 + 16] = 65535; // Natural AC index 2 has a scale overflow.
    capture
        .set_quantization(&QuantEncoding::Raw {
            qtable: quantization.into(),
            qtable_den: 1.0 / 2040.0,
        })
        .unwrap();
    let mut data = capture.new_group(&header, 0).unwrap();
    let mut raw = [[0; 64]; 3];
    raw[0][8] = 4096;
    assert_eq!(
        ac_error_reason(
            data.capture_block(&header, 0, 0, 0, raw.each_ref(), [127, 0, 0])
                .unwrap_err()
        ),
        "JPEG capture AC coefficient is outside the 8-bit JPEG range",
    );
    raw[0][8] = 17;
    assert_eq!(
        ac_error_reason(
            data.capture_block(&header, 0, 0, 0, raw.each_ref(), [127, 0, 0])
                .unwrap_err()
        ),
        "JPEG capture CfL scale overflow",
    );
    assert_eq!(data.coefficients[0][1], 17);
    assert_eq!(data.coefficients[0][2], 0);
    data.capture_block(&header, 0, 0, 0, raw.each_ref(), [0; 3])
        .unwrap();
    // Returning to a nonzero factor must not reuse the bypass's zero result.
    assert_eq!(
        ac_error_reason(
            data.capture_block(&header, 0, 0, 0, raw.each_ref(), [127, 0, 0])
                .unwrap_err()
        ),
        "JPEG capture CfL scale overflow",
    );
}

#[test]
fn captures_dc_ac_tables_and_subsampled_group_coordinates() {
    for (width, height, ycbcr, modes, gray) in [
        (17, 19, true, [0, 0, 0], true),
        (17, 19, false, [0, 0, 0], false),
        (2057, 19, true, [0, 0, 0], false),
        (19, 2057, true, [0, 1, 0], false),
        (257, 259, true, [0, 2, 0], false),
        (257, 259, true, [0, 3, 0], false),
        (17, 19, true, [1, 1, 1], true),
    ] {
        let header = header(width, height, ycbcr, modes);
        let capture = JpegCapture::new(&header, gray).unwrap();
        capture.set_quantization(&tables()).unwrap();
        assert!(capture.take().is_none());
        assert!(capture.finish().is_err());
        for group in (0..header.num_lf_groups()).rev() {
            let rect = header.lf_group_rect(group);
            let mut input: [Image<i32>; 3] = std::array::from_fn(|m| {
                let c = [1, 0, 2][m];
                Image::new((
                    rect.size.0 >> header.hshift(c),
                    rect.size.1 >> header.vshift(c),
                ))
                .unwrap()
            });
            for (m, plane) in input.iter_mut().enumerate() {
                let c = [1, 0, 2][m];
                let offset = if ycbcr { 0 } else { 1024 / (3 * c as i32 + 1) };
                for y in 0..plane.size().1 {
                    for (x, value) in plane.row_mut(y).iter_mut().enumerate() {
                        *value = dc(
                            c,
                            (rect.origin.0 >> header.hshift(c)) + x,
                            (rect.origin.1 >> header.vshift(c)) + y,
                        ) + offset;
                    }
                }
            }
            let raw = input.map(Image::into_raw);
            capture
                .capture_lf(
                    &header,
                    group,
                    [&raw[0], &raw[1], &raw[2]].map(|p| p.as_rect()),
                    ModularStorage::I32,
                    0,
                )
                .unwrap();
        }
        for group in (0..header.num_groups()).rev() {
            let rect = header.block_group_rect(group);
            let mut data = capture.new_group(&header, group).unwrap();
            for by in 0..rect.size.1 {
                for bx in 0..rect.size.0 {
                    let raw = std::array::from_fn(|c| {
                        std::array::from_fn(|k| {
                            ac(
                                c,
                                (rect.origin.0 + bx) >> header.hshift(c),
                                (rect.origin.1 + by) >> header.vshift(c),
                                (k % 8) * 8 + k / 8,
                            )
                        })
                    });
                    data.capture_block(&header, group, bx, by, raw.each_ref(), [0; 3])
                        .unwrap();
                }
            }
            capture.store_group(&header, group, data).unwrap();
        }
        assert!(capture.take().is_none());
        capture.finish().unwrap();
        let result = capture.take().unwrap();
        assert!(capture.take().is_none());
        assert_eq!((result.width, result.height), (width, height));
        let channels: &[usize] = if gray {
            &[1]
        } else if ycbcr {
            &[1, 0, 2]
        } else {
            &[0, 1, 2]
        };
        assert_eq!(
            result
                .components
                .iter()
                .map(|p| p.jxl_channel)
                .collect::<Vec<_>>(),
            channels
        );
        for component in result.components {
            let c = component.jxl_channel;
            assert_eq!(
                component.width_in_blocks,
                header.size_blocks().0 >> header.hshift(c)
            );
            assert_eq!(
                component.height_in_blocks,
                header.size_blocks().1 >> header.vshift(c)
            );
            assert_eq!(
                component.horizontal_sampling_factor,
                1 << header.raw_hshift(c)
            );
            assert_eq!(
                component.vertical_sampling_factor,
                1 << header.raw_vshift(c)
            );
            assert_eq!(
                component.quantization,
                std::array::from_fn(|k| (3 * c + k + 1) as u16)
            );
            for y in 0..component.height_in_blocks {
                for x in 0..component.width_in_blocks {
                    let block =
                        &component.coefficients[(y * component.width_in_blocks + x) * 64..][..64];
                    assert_eq!(i32::from(block[0]), dc(c, x, y));
                    for k in 1..64 {
                        assert_eq!(i32::from(block[k]), ac(c, x, y, k));
                    }
                }
            }
        }
    }
}

#[test]
fn cfl_uses_signed_truncation_and_two_fixed_point_roundings() {
    for factor in [-128, -84, -1, 0, 1, 42, 84, 127] {
        for luma in [-199, -3, -1, 0, 1, 3, 199] {
            for (q_y, q_c) in [(1, 1), (3, 7), (7, 3), (65535, 65535)] {
                let ratio = 2048i64 * i64::from(q_y) / i64::from(q_c);
                let scale = i64::from(factor) * 2048 / 84;
                let scale = (ratio * scale + 1024) >> 11;
                let prediction = (i64::from(luma) * scale + 1024) >> 11;
                assert_eq!(
                    i64::from(restore_ac(-17, luma, q_y, q_c, factor).unwrap()),
                    -17 + prediction
                );
            }
        }
    }
    assert!(restore_ac(4096, 0, 1, 1, 0).is_err());
    assert!(restore_ac(-4096, 0, 1, 1, 0).is_err());
    assert!(restore_ac(0, i32::MAX, 1, 1, 84).is_err());
    assert!(restore_ac(0, 1, 65535, 1, 127).is_err());
}

#[test]
fn group_cfl_transposes_once_and_does_not_modify_input() {
    let header = header(8, 8, true, [0; 3]);
    let capture = JpegCapture::new(&header, false).unwrap();
    capture
        .set_quantization(&QuantEncoding::Raw {
            qtable: vec![7; 192],
            qtable_den: 1.0 / 2040.0,
        })
        .unwrap();
    let mut group = capture.new_group(&header, 0).unwrap();
    let raw: [[i32; 64]; 3] =
        std::array::from_fn(|c| std::array::from_fn(|k| (c * 100 + k) as i32 - 140));
    let original = raw;
    group
        .capture_block(&header, 0, 0, 0, raw.each_ref(), [84, 0, -84])
        .unwrap();
    for k in 1..64 {
        let t = (k % 8) * 8 + k / 8;
        assert_eq!(i32::from(group.coefficients[0][k]), raw[0][t] + raw[1][t]);
        assert_eq!(i32::from(group.coefficients[1][k]), raw[1][t]);
        assert_eq!(i32::from(group.coefficients[2][k]), raw[2][t] - raw[1][t]);
    }
    assert_eq!(raw, original);
}

#[test]
fn rejects_non_jpeg_quantization_and_correlation() {
    let header = header(8, 8, true, [0; 3]);
    let capture = JpegCapture::new(&header, false).unwrap();
    assert!(capture.set_quantization(&QuantEncoding::Library).is_err());
    for (entry, denominator) in [
        (0, 1.0 / 2040.0),
        (65536, 1.0 / 2040.0),
        (1, 0.5),
        (1, f32::NAN),
    ] {
        assert!(
            capture
                .set_quantization(&QuantEncoding::Raw {
                    qtable: vec![entry; 192],
                    qtable_den: denominator
                })
                .is_err()
        );
    }
    capture
        .set_quantization(&QuantEncoding::Raw {
            qtable: vec![65535; 192],
            qtable_den: 1.0 / 2040.0,
        })
        .unwrap();
    let mut params = ColorCorrelationParams::default();
    assert!(capture.validate_lf_params(&params).is_err());
    params.base_correlation_b = 0.0;
    capture.validate_lf_params(&params).unwrap();
    params.ytox_lf = 1;
    assert!(capture.validate_lf_params(&params).is_err());
    params.ytox_lf = 0;
    params.ytob_lf = -1;
    assert!(capture.validate_lf_params(&params).is_err());
    params.ytob_lf = 0;
    params.color_factor += 1;
    assert!(capture.validate_lf_params(&params).is_err());
    let input: [Image<i16>; 3] = std::array::from_fn(|_| Image::new_with_value((1, 1), 7).unwrap());
    let input = input.map(Image::into_raw);
    let raw = [&input[0], &input[1], &input[2]].map(|p| p.as_rect());
    assert!(
        capture
            .capture_lf(&header, 0, raw, ModularStorage::I16, 1)
            .is_err()
    );
    capture
        .capture_lf(&header, 0, raw, ModularStorage::I16, 0)
        .unwrap();
    assert_eq!(
        capture
            .state
            .lock()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .components[0]
            .coefficients[0],
        7
    );
}

#[test]
fn completion_requires_every_group_and_rejects_dc_clamping() {
    let header = header(8, 8, true, [0; 3]);
    let capture = JpegCapture::new(&header, true).unwrap();
    capture.set_quantization(&tables()).unwrap();
    assert!(capture.finish().is_err());
    let input: [Image<i32>; 3] =
        std::array::from_fn(|_| Image::new_with_value((1, 1), 2048).unwrap());
    capture
        .capture_lf_impl(
            &header,
            0,
            input.each_ref().map(|p| {
                p.get_rect(Rect {
                    origin: (0, 0),
                    size: (1, 1),
                })
            }),
            0,
        )
        .unwrap();
    let group = capture.new_group(&header, 0).unwrap();
    capture.store_group(&header, 0, group).unwrap();
    assert!(capture.finish().is_err());
    assert!(capture.take().is_none());
    capture.abort();
    assert!(capture.take().is_none());
    assert!(capture.new_group(&header, 0).is_err());
}

#[test]
fn capture_handle_outlives_frame_and_has_single_take() {
    let header = header(8, 8, true, [0; 3]);
    let inner = Arc::new(JpegCapture::new(&header, true).unwrap());
    let handle = JxlJpegCoefficientCapture {
        inner: inner.clone(),
    };
    inner.set_quantization(&tables()).unwrap();
    let input: [Image<i16>; 3] =
        std::array::from_fn(|_| Image::new_with_value((1, 1), -13).unwrap());
    inner
        .capture_lf_impl(
            &header,
            0,
            input.each_ref().map(|p| {
                p.get_rect(Rect {
                    origin: (0, 0),
                    size: (1, 1),
                })
            }),
            0,
        )
        .unwrap();
    inner
        .store_group(&header, 0, inner.new_group(&header, 0).unwrap())
        .unwrap();
    inner.finish().unwrap();
    inner.abort(); // A reset after successful completion preserves the detached result.
    drop(inner);
    assert_eq!(handle.take().unwrap().components[0].coefficients[0], -13);
    assert!(handle.take().is_none());
}

#[test]
fn extra_dc_precision_requires_exact_signed_integer_coefficients() {
    for ycbcr in [false, true] {
        let header = header(8, 8, ycbcr, [0; 3]);
        for precision in 0..=3 {
            for value in [-2047, -1, 0, 1, 2047] {
                let capture = JpegCapture::new(&header, false).unwrap();
                capture.set_quantization(&tables()).unwrap();
                let input: [Image<i32>; 3] = std::array::from_fn(|m| {
                    let c = [1, 0, 2][m];
                    let offset = if ycbcr { 0 } else { 1024 / (3 * c as i32 + 1) };
                    Image::new_with_value((1, 1), (value + offset) << precision).unwrap()
                });
                let input = input.map(Image::into_raw);
                let raw = [&input[0], &input[1], &input[2]].map(|p| p.as_rect());
                capture
                    .capture_lf(&header, 0, raw, ModularStorage::I32, precision)
                    .unwrap();
                capture
                    .store_group(&header, 0, capture.new_group(&header, 0).unwrap())
                    .unwrap();
                capture.finish().unwrap();
                for component in capture.take().unwrap().components {
                    assert_eq!(i32::from(component.coefficients[0]), value);
                }
            }
        }
    }
    let header = header(8, 8, true, [0; 3]);
    for value in [-17, -1, 1, 17] {
        let capture = JpegCapture::new(&header, true).unwrap();
        let input: [Image<i32>; 3] =
            std::array::from_fn(|_| Image::new_with_value((1, 1), value).unwrap());
        let input = input.map(Image::into_raw);
        let raw = [&input[0], &input[1], &input[2]].map(|p| p.as_rect());
        for precision in 1..=4 {
            assert!(
                capture
                    .capture_lf(&header, 0, raw, ModularStorage::I32, precision)
                    .is_err()
            );
        }
    }
}

#[test]
fn multipass_capture_rejects_missing_duplicate_and_reordered_passes() {
    let mut header = header(8, 8, true, [0; 3]);
    header.passes.num_passes = 3;
    header.passes.shift = vec![2, 1];
    let capture = JpegCapture::new(&header, false).unwrap();
    capture.set_quantization(&tables()).unwrap();
    assert!(
        capture
            .validate_passes(&header, 0, [1].into_iter())
            .is_err()
    );
    assert!(
        capture
            .validate_passes(&header, 0, [0, 0].into_iter())
            .is_err()
    );
    assert!(
        capture
            .validate_passes(&header, 0, [0, 2].into_iter())
            .is_err()
    );
    assert!(
        !capture
            .validate_passes(&header, 0, [0].into_iter())
            .unwrap()
    );
    capture.record_passes(0, 1);
    assert!(
        capture
            .validate_passes(&header, 0, [0].into_iter())
            .is_err()
    );
    assert!(
        capture
            .validate_passes(&header, 0, [1, 2, 3].into_iter())
            .is_err()
    );
    assert!(
        capture
            .validate_passes(&header, 0, [1, 2].into_iter())
            .unwrap()
    );
    assert!(capture.finish().is_err());
    assert!(capture.take().is_none());
    let mut data = capture.new_group(&header, 0).unwrap();
    let mut accumulated = [[0i32; 64]; 3];
    // The first pass is not representable as JPEG; the complete sum is.
    // CfL rounding must happen once on that sum, never on individual passes.
    for (value, shift) in [(1100, 2), (-2198, 1), (-3, 0)] {
        accumulated[1][8] += value << shift;
    }
    assert_eq!(accumulated[1][8], 1);
    data.capture_block(&header, 0, 0, 0, accumulated.each_ref(), [42, 0, -42])
        .unwrap();
    assert_eq!(data.coefficients[1][1], 1);
    capture.store_group(&header, 0, data).unwrap();
    capture.record_passes(0, 2);
    assert!(
        capture
            .validate_passes(&header, 0, [2].into_iter())
            .is_err()
    );
    assert!(capture.take().is_none()); // Still missing LF.
}

#[test]
fn jpeg_groups_accumulate_without_a_pixel_pipeline_or_idct_scratch() {
    use crate::entropy_coding::decode::Histograms;
    use crate::frame::block_context_map::BlockContextMap;
    use crate::frame::quant_weights::DequantMatrices;
    use crate::frame::quantizer::{LfQuantFactors, QuantizerParams};
    use crate::frame::{DecoderState, HfGlobalState, LfGlobalState, PassState};
    use crate::headers::permutation::Permutation;
    use crate::headers::toc::Toc;
    use crate::headers::{FileHeader, JxlHeader};
    use crate::image::BufferRecycler;

    struct Runner;
    impl JxlParallelRunner for Runner {
        fn run(&mut self, num: usize, fun: &crate::api::JxlParallelRunnerFun<'_>) -> Result<()> {
            for i in 0..num {
                fun(i)?;
            }
            Ok(())
        }
        fn num_threads(&self) -> usize {
            1
        }
    }

    fn singleton_histogram(contexts: usize, nonzero: bool) -> Histograms {
        // No LZ77; all contexts use one prefix histogram; direct uint tokens;
        // singleton 0 means no nonzeros, singleton 1 means one value of -1.
        let mut fields = vec![(1, 0), (1, 1), (2, 0), (1, 1), (4, 15)];
        if nonzero {
            fields.extend([(1, 1), (4, 0), (2, 1), (2, 0), (1, 1)]);
        } else {
            fields.push((1, 0)); // Alphabet size 1 needs no Huffman tree.
        }
        let bits: Vec<u8> = fields
            .into_iter()
            .flat_map(|(n, value)| (0..n).map(move |i| ((value >> i) & 1) as u8))
            .collect();
        let bytes: Vec<u8> = bits
            .chunks(8)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0, |v, (i, bit)| v | (*bit << i))
            })
            .collect();
        Histograms::decode(contexts, &mut BitReader::new(&bytes), true, &mut vec![]).unwrap()
    }

    for (batches, nonzeros) in [vec![3], vec![1, 2], vec![2, 1], vec![1, 1, 1]]
        .into_iter()
        .flat_map(|batches| {
            [[true; 3], [true, false, false], [false; 3]]
                .map(|nonzeros| (batches.clone(), nonzeros))
        })
    {
        // 8x8 default file header; replace XYB metadata for the JPEG test frame.
        let mut file = FileHeader::read(&mut BitReader::new(&[0xff, 0x0a, 0x41, 0x06])).unwrap();
        file.image_metadata.xyb_encoded = false;
        let mut header = header(8, 8, true, [0; 3]);
        header.passes.num_passes = 3;
        header.passes.shift = vec![2, 1];
        let recycler = Arc::new(BufferRecycler::default());
        let modular_global = super::super::modular::FullModularImage::read(
            &header,
            &file.image_metadata,
            0,
            &mut BitReader::new(&[]),
            recycler.clone(),
            None,
            ModularStorage::I32,
            true,
        )
        .unwrap();
        let block_context_map =
            BlockContextMap::read(&mut BitReader::new(&[1]), &mut vec![]).unwrap();
        let contexts = block_context_map.num_ac_contexts() + 16;
        let toc = Toc {
            permuted: false,
            permutation: Permutation::default(),
            entries: vec![],
        };
        let state = DecoderState::new(file, &Default::default(), true);
        let mut frame = Frame::from_header_and_toc(header, toc, state, (1, 0), recycler).unwrap();
        let handle = frame.enable_jpeg_coefficient_capture().unwrap();
        let capture = handle.inner.clone();
        capture
            .set_quantization(&QuantEncoding::Raw {
                qtable: vec![7; 192],
                qtable_den: 1.0 / 2040.0,
            })
            .unwrap();
        let input: [Image<i32>; 3] =
            std::array::from_fn(|_| Image::new_with_value((1, 1), -24).unwrap());
        let input = input.map(Image::into_raw);
        capture
            .capture_lf(
                &frame.header,
                0,
                [&input[0], &input[1], &input[2]].map(|p| p.as_rect()),
                ModularStorage::I32,
                3,
            )
            .unwrap();
        frame.lf_global = Some(LfGlobalState {
            lf_quant: LfQuantFactors::default(),
            quant_params: Some(QuantizerParams {
                global_scale: 17,
                quant_lf: 9,
            }),
            block_context_map: Some(block_context_map),
            color_correlation_params: Some(ColorCorrelationParams {
                base_correlation_b: 0.0,
                ..Default::default()
            }),
            tree: None,
            modular_global,
            total_bits_read: 0,
        });
        let dequant_matrices = DequantMatrices::decode(
            &frame.header,
            frame.lf_global.as_ref().unwrap(),
            &mut BitReader::new(&[1]),
            None,
        )
        .unwrap();
        frame.hf_global = Some(HfGlobalState {
            num_histograms: 1,
            passes: nonzeros
                .into_iter()
                .map(|nonzero| PassState {
                    coeff_orders: vec![Permutation((0..64).collect::<Vec<_>>().into()); 3],
                    histograms: singleton_histogram(contexts, nonzero),
                })
                .collect(),
            dequant_matrices,
            hf_coefficients: vec![Mutex::new(vec![])],
            use_i16: false,
        });
        let meta = frame.hf_meta.as_mut().unwrap();
        meta.transform_map.row_mut(0)[0] = 128; // DCT8, first block.
        meta.raw_quant_map.row_mut(0)[0] = 39;
        meta.ytox_map.row_mut(0)[0] = 42;
        meta.ytob_map.row_mut(0)[0] = -42;
        {
            let mut buffers = frame.buffer_recycler.vardct_buffers.get();
            buffers.coeffs_storage.resize(
                crate::util::num_cache_lines_for::<i32>(192),
                Default::default(),
            );
            crate::util::slice_from_cachelines_mut::<i32>(&mut buffers.coeffs_storage).fill(12345);
        }
        let mut next = 0;
        for count in batches {
            let passes = (next..next + count)
                .map(|pass| (pass, BitReader::new(&[])))
                .collect();
            frame
                .decode_jpeg_groups(&mut [(0, passes)], &mut Runner)
                .unwrap();
            next += count;
            if next != 3 {
                assert!(capture.finish().is_err());
            }
        }
        capture.finish().unwrap();
        assert!(frame.render_pipeline.is_none());
        frame.finalize_lf(&mut Runner).unwrap();
        let buffers = frame.buffer_recycler.vardct_buffers.get();
        assert!(buffers.scratch.is_empty());
        assert!(buffers.transform_buffer.iter().all(Vec::is_empty));
        assert!(buffers.lf_upsample_bufs.is_none());
        let output = handle.take().unwrap();
        let y = -(i32::from(nonzeros[0]) * 4 + i32::from(nonzeros[1]) * 2 + i32::from(nonzeros[2]));
        let expected = [y, y + ((y + 1) >> 1), y + ((-y + 1) >> 1)];
        for (component, ac) in output.components.iter().zip(expected) {
            assert_eq!(component.coefficients[0], -3);
            assert_eq!(i32::from(component.coefficients[8]), ac);
            assert!(
                component
                    .coefficients
                    .iter()
                    .enumerate()
                    .all(|(i, v)| i == 0 || i == 8 || *v == 0)
            );
        }
    }
}

#[test]
fn rgb_finish_is_transactional_and_removes_dc_offset_only_once() {
    let header = header(8, 8, false, [0; 3]);
    let capture = JpegCapture::new(&header, false).unwrap();
    capture.set_quantization(&tables()).unwrap();
    capture
        .store_group(&header, 0, capture.new_group(&header, 0).unwrap())
        .unwrap();
    for invalid in [true, false] {
        let input: [Image<i32>; 3] = std::array::from_fn(|m| {
            let c = [1, 0, 2][m];
            let value = if invalid && c == 1 { 2048 } else { 17 };
            Image::new_with_value((1, 1), value + 1024 / (3 * c as i32 + 1)).unwrap()
        });
        let input = input.map(Image::into_raw);
        capture
            .capture_lf(
                &header,
                0,
                [&input[0], &input[1], &input[2]].map(|p| p.as_rect()),
                ModularStorage::I32,
                0,
            )
            .unwrap();
        if invalid {
            assert!(capture.finish().is_err());
            assert!(capture.take().is_none());
            let state = capture.state.lock().unwrap();
            assert_eq!(
                state.output.as_ref().unwrap().components[0].coefficients[0],
                1041
            );
        }
    }
    capture.finish().unwrap();
    capture.finish().unwrap();
    capture.abort(); // A later reset/error cannot alter a completed frame.
    let output = capture.take().unwrap();
    assert!(
        output
            .components
            .iter()
            .all(|component| component.coefficients[0] == 17)
    );
    assert!(capture.take().is_none());
}
