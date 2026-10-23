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
                    data.capture_block(&header, group, bx, by, &raw, [0; 3])
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
        .capture_block(&header, 0, 0, 0, &raw, [84, 0, -84])
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
fn rejects_non_jpeg_quantization_correlation_and_precision() {
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
    assert!(capture.validate_lf_params(&params, 65536, 1).is_err());
    params.base_correlation_b = 0.0;
    capture.validate_lf_params(&params, 65536, 1).unwrap();
    assert!(capture.validate_lf_params(&params, 1, 1).is_err());
    assert!(capture.validate_lf_params(&params, 65536, 2).is_err());
    params.ytox_lf = 1;
    assert!(capture.validate_lf_params(&params, 65536, 1).is_err());
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
