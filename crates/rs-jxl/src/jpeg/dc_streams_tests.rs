//! Small preparation fixtures and independent scalar arithmetic, not a codec.

use super::*;
use crate::jpeg::data::{
    CodingProcess, Component, ComponentScanInfo, JpegData, QuantTable, ScanInfo,
};

fn fixture(width: u32, height: u32, sampling: &[(u32, u32)], quant: &[i32]) -> JpegData {
    let max_h = sampling.iter().map(|s| s.0).max().unwrap();
    let max_v = sampling.iter().map(|s| s.1).max().unwrap();
    let components = sampling
        .iter()
        .enumerate()
        .map(|(c, &(h, v))| {
            let width_in_blocks = width.div_ceil(8 * max_h) * h;
            let height_in_blocks = height.div_ceil(8 * max_v) * v;
            let mut coefficients =
                vec![0; width_in_blocks as usize * height_in_blocks as usize * 64];
            for (block, values) in coefficients.chunks_exact_mut(64).enumerate() {
                values[0] = ((block + 19 * c) % 83) as i16 - 41;
            }
            Component {
                id: c as u32 + 1,
                h_samp_factor: h,
                v_samp_factor: v,
                quant_index: c as u32,
                width_in_blocks,
                height_in_blocks,
                coefficients,
            }
        })
        .collect();
    JpegData {
        width,
        height,
        precision: 8,
        coding_process: Some(CodingProcess::Baseline),
        components,
        quant: quant
            .iter()
            .enumerate()
            .map(|(index, &value)| QuantTable {
                index: index as u32,
                precision: u32::from(value > 255),
                values: [value; 64],
                is_last: true,
            })
            .collect(),
        scan_info: vec![ScanInfo {
            components: (0..sampling.len())
                .map(|component_index| ComponentScanInfo {
                    component_index,
                    ..Default::default()
                })
                .collect(),
            ss: 0,
            se: 63,
            quant_table_indices: std::array::from_fn(|i| (i < quant.len()).then_some(i)),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn geometry(jpeg: &JpegData, layout: &TranscodeLayout<'_>) -> FrameGeometry {
    FrameGeometry::new(jpeg.width, jpeg.height, layout.subsampling()).unwrap()
}

fn quantization(layout: &TranscodeLayout<'_>) -> DcQuantization {
    DcQuantization::from_layout(layout.planes().each_ref().map(|p| p.quantization())).unwrap()
}

fn dc(jpeg: &JpegData, group: usize) -> Result<DcGroup> {
    let layout = TranscodeLayout::new(jpeg).unwrap();
    prepare_dc_group(
        &layout,
        &quantization(&layout),
        &layout.dc_contexts(),
        geometry(jpeg, &layout),
        group,
        &PrepareLimits::default(),
    )
}

#[test]
fn source_f32_steps_and_modular_y_x_swap_match_scalar_reference() {
    let jpeg = fixture(24, 16, &[(1, 1); 3], &[3, 7, 13]);
    let layout = TranscodeLayout::new(&jpeg).unwrap();
    let rounded = quantization(&layout);
    let prepared = dc(&jpeg, 0).unwrap();
    assert_eq!(prepared.extra_dc_precision, 0);
    assert_eq!(JPEG_STREAM_BITS, 8);
    assert_eq!(
        prepared.rect,
        BlockRect {
            x: 0,
            y: 0,
            width: 3,
            height: 2
        }
    );
    for (modular, internal) in [1, 0, 2].into_iter().enumerate() {
        let plane = &layout.planes()[internal];
        let component = &jpeg.components[plane.jpeg_component()];
        let q = jpeg.quant[component.quant_index as usize].values[0];
        let original_scale = 1.0f32 / (255.0f32 * 8.0 / q as f32);
        let channel = &prepared.channels[modular];
        assert_eq!(
            (
                channel.width,
                channel.height,
                channel.hshift,
                channel.vshift
            ),
            (3, 2, 0, 0)
        );
        for (index, &encoded) in channel.samples.iter().enumerate() {
            let original = i32::from(component.coefficients[index * 64]);
            let scaled = original as f32 * original_scale;
            let expected = (scaled * rounded.inverse_dc_quants()[internal]).round() as i32;
            assert_eq!(encoded, expected);
            assert_eq!(encoded, original);
        }
    }
    assert_eq!(prepared.channels[0].samples[0], -41);
    assert_eq!(prepared.channels[1].samples[0], -22);
    assert_eq!(prepared.channels[2].samples[0], -3);
}

#[test]
fn rgb_integer_offset_is_applied_before_the_original_float_scale() {
    let mut jpeg = fixture(8, 8, &[(1, 1); 3], &[1, 3, 7]);
    for (component, id) in jpeg.components.iter_mut().zip(b"RGB") {
        component.id = u32::from(*id);
        component.coefficients[0] = -10;
    }
    let before = jpeg.components.clone();
    let prepared = dc(&jpeg, 0).unwrap();
    // RGB JpegOrder is 0,1,2; Modular still swaps internal channels 0 and 1.
    assert_eq!(prepared.channels[0].samples, [331]);
    assert_eq!(prepared.channels[1].samples, [1014]);
    assert_eq!(prepared.channels[2].samples, [136]);
    assert_eq!(jpeg.components, before);
}

#[test]
fn grayscale_chroma_is_zero_including_all_padded_mcu_edge_blocks() {
    let mut jpeg = fixture(1, 1, &[(2, 2)], &[255]);
    for (block, value) in jpeg.components[0]
        .coefficients
        .chunks_exact_mut(64)
        .zip([-7, 2, 3, 9])
    {
        block[0] = value;
    }
    let layout = TranscodeLayout::new(&jpeg).unwrap();
    assert_eq!(layout.subsampling().channel_modes(), [1; 3]);
    assert_eq!(layout.subsampling().horizontal_shifts(), [0; 3]);
    assert_eq!(layout.subsampling().vertical_shifts(), [0; 3]);
    assert_eq!(geometry(&jpeg, &layout).subsampling(), layout.subsampling());
    for plane in layout.planes() {
        assert_eq!(plane.dimensions_in_blocks(), (2, 2));
    }
    let prepared = dc(&jpeg, 0).unwrap();
    assert_eq!((prepared.rect.width, prepared.rect.height), (2, 2));
    assert_eq!(prepared.channels[0].samples, [-7, 2, 3, 9]);
    for chroma in &prepared.channels[1..] {
        assert_eq!((chroma.width, chroma.height), (2, 2));
        assert_eq!(chroma.samples, [0; 4]);
    }
    assert_eq!(prepared.quant_dc.len(), 4);
}

#[test]
fn subsampled_rectangles_and_group_offsets_use_padded_frame_geometry() {
    for sampling in [
        [(2, 2), (1, 1), (1, 1)],
        [(2, 1), (1, 1), (1, 1)],
        [(1, 2), (1, 1), (1, 1)],
        [(1, 1), (2, 2), (1, 1)],
    ] {
        for (width, height) in [(2051, 17), (17, 2051)] {
            let jpeg = fixture(width, height, &sampling, &[1; 3]);
            let layout = TranscodeLayout::new(&jpeg).unwrap();
            let geometry = geometry(&jpeg, &layout);
            assert_eq!(geometry.dc_group_count(), 2);
            for group in 0..2 {
                let prepared = dc(&jpeg, group).unwrap();
                let r = geometry.dc_group_rect(group).unwrap();
                let h = layout.subsampling().horizontal_shifts();
                let v = layout.subsampling().vertical_shifts();
                for (modular, c) in [1, 0, 2].into_iter().enumerate() {
                    let channel = &prepared.channels[modular];
                    let plane = &layout.planes()[c];
                    assert_eq!(
                        (channel.width, channel.height),
                        (r.width >> h[c], r.height >> v[c])
                    );
                    assert_eq!((channel.hshift, channel.vshift), (0, 0));
                    for y in 0..channel.height {
                        for x in 0..channel.width {
                            assert_eq!(
                                channel.samples[y * channel.width + x],
                                plane
                                    .adjusted_dc(
                                        ((r.x >> h[c]) + x) as u32,
                                        ((r.y >> v[c]) + y) as u32
                                    )
                                    .unwrap()
                            );
                        }
                    }
                }
                let thresholds = layout.dc_contexts();
                for y in 0..r.height {
                    for x in 0..r.width {
                        let channel = &prepared.channels[0];
                        let encoded = channel.samples[(y >> v[1]) * channel.width + (x >> h[1])];
                        let expected = thresholds
                            .luma_thresholds()
                            .iter()
                            .filter(|&&t| encoded > t)
                            .count() as u8;
                        assert_eq!(prepared.quant_dc[y * r.width + x], expected);
                    }
                }
            }
        }
    }
}

#[test]
fn sixteen_bit_quantization_samples_preserve_small_dc_via_source_f16_factors() {
    // A small fixed image per iteration; no dimension-dependent test allocation.
    let mut jpeg = fixture(8, 8, &[(1, 1)], &[1]);
    jpeg.components[0].coefficients[0] = -17;
    for value in [1, 7, 31, 37, 127, 255, 256, 511, 1024, 32768, 65535] {
        jpeg.quant[0].precision = u32::from(value > 255);
        jpeg.quant[0].values = [value; 64];
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let quant = quantization(&layout);
        let original = -17.0f32 * (1.0f32 / (2040.0f32 / value as f32));
        let expected = (original * quant.inverse_dc_quants()[1]).round() as i32;
        let prepared = dc(&jpeg, 0).unwrap();
        assert_eq!(prepared.channels[0].samples, [expected]);
        assert_eq!(expected, -17);
    }
}

#[test]
fn source_float_rounding_is_not_replaced_by_the_original_integer() {
    let mut jpeg = fixture(8, 8, &[(1, 1)], &[37]);
    for original in [-1024, 1023] {
        jpeg.components[0].coefficients[0] = original;
        let expected = if original < 0 { -1025 } else { 1024 };
        assert!(matches!(dc(&jpeg, 0), Err(DcStreamError::DcNotReversible {
            channel: 1, x: 0, y: 0, adjusted, encoded,
        }) if adjusted == i32::from(original) && encoded == expected));
    }
    // A bypass using the original integer would silently diverge from source.
    assert_eq!(
        encoded_dc(
            1023,
            1.0f32 / (2040.0f32 / 37.0),
            quantization(&TranscodeLayout::new(&jpeg).unwrap()).inverse_dc_quants()[1]
        )
        .unwrap(),
        1024
    );
}

#[test]
fn parsed_i16_extremes_and_source_clamping_are_explicit_errors() {
    let mut jpeg = fixture(8, 8, &[(1, 1)], &[255]);
    for original in [i16::MIN, -2048, 2048, i16::MAX] {
        jpeg.components[0].coefficients[0] = original;
        assert!(matches!(dc(&jpeg, 0), Err(DcStreamError::DcOutOfRange {
            channel: 1, x: 0, y: 0, coefficient,
        }) if coefficient == i32::from(original)));
    }
    for original in [-2047, -1024, 0, 1023, 2047] {
        jpeg.components[0].coefficients[0] = original;
        assert_eq!(
            dc(&jpeg, 0).unwrap().channels[0].samples,
            [i32::from(original)]
        );
    }
}

#[test]
fn scalar_round_is_ties_away_from_zero_and_finite_i32_is_checked() {
    for (value, expected) in [(-3, -2), (-1, -1), (1, 1), (3, 2)] {
        assert_eq!(encoded_dc(value, 0.5, 1.0).unwrap(), expected);
    }
    for (scale, inverse) in [(f32::NAN, 1.0), (1.0, f32::INFINITY), (f32::MAX, 2.0)] {
        assert!(encoded_dc(1, scale, inverse).is_err());
    }
    assert!(encoded_dc(i32::MAX, 1.0, 1.0).is_err());
    assert_eq!(encoded_dc(i32::MIN, 1.0, 1.0).unwrap(), i32::MIN);
}

#[test]
fn qdc_uses_encoded_integer_samples_with_strict_thresholds_and_replication() {
    let channel = Channel {
        width: 2,
        height: 2,
        hshift: 0,
        vshift: 0,
        samples: vec![-2, -1, 0, 10],
    };
    let rect = BlockRect {
        x: 256,
        y: 256,
        width: 4,
        height: 4,
    };
    let mut qdc = vec![255; 16];
    luma_buckets(&channel, rect, (1, 1), &[-1, 0, 10], &mut qdc);
    assert_eq!(qdc, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2]);
    luma_buckets(&channel, rect, (1, 1), &[], &mut qdc);
    assert_eq!(qdc, [0; 16]);
}

#[test]
fn ac_metadata_tiles_flattened_rows_and_epf_match_source_shapes() {
    for (frame_width, frame_height) in [(2051, 17), (17, 2051)] {
        let jpeg = fixture(
            frame_width,
            frame_height,
            &[(2, 2), (1, 1), (1, 1)],
            &[1; 3],
        );
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let geometry = geometry(&jpeg, &layout);
        let (blocks_x, blocks_y) = geometry.dimensions_in_blocks();
        let width = blocks_x.div_ceil(8);
        let height = blocks_y.div_ceil(8);
        let ytox: Vec<i8> = (0..width * height).map(|i| (i % 256) as u8 as i8).collect();
        let ytob: Vec<i8> = (0..width * height).map(|i| -1 - (i % 127) as i8).collect();
        for group in 0..geometry.dc_group_count() {
            let prepared = prepare_ac_metadata_group(
                geometry,
                group,
                CflMaps {
                    width,
                    height,
                    ytox: &ytox,
                    ytob: &ytob,
                },
                &PrepareLimits::default(),
            )
            .unwrap();
            let r = prepared.rect;
            let tiles_x = r.width.div_ceil(8);
            let tiles_y = r.height.div_ceil(8);
            for (c, map) in [&ytox, &ytob].into_iter().enumerate() {
                let channel = &prepared.channels[c];
                assert_eq!(
                    (
                        channel.width,
                        channel.height,
                        channel.hshift,
                        channel.vshift
                    ),
                    (tiles_x, tiles_y, 3, 3)
                );
                for y in 0..tiles_y {
                    for x in 0..tiles_x {
                        assert_eq!(
                            channel.samples[y * tiles_x + x],
                            i32::from(map[(r.y / 8 + y) * width + r.x / 8 + x])
                        );
                    }
                }
            }
            let blocks = r.width * r.height;
            assert_eq!(prepared.ac_metadata_size, blocks);
            assert_eq!(
                (prepared.channels[2].width, prepared.channels[2].height),
                (blocks, 2)
            );
            assert_eq!(
                (prepared.channels[3].width, prepared.channels[3].height),
                (r.width, r.height)
            );
            for channel in &prepared.channels[2..] {
                assert_eq!((channel.hshift, channel.vshift), (0, 0));
                assert!(channel.samples.iter().all(|&v| v == 0));
            }
        }
    }
}

#[test]
fn default_preparation_budget_checks_only_numeric_overflow() {
    let limits = PrepareLimits::default();
    assert!(check_budget(1_048_577, 0, &limits).is_ok());
    assert!(matches!(
        check_budget(usize::MAX, 1, &limits),
        Err(DcStreamError::Limit("output sample count overflow"))
    ));
    assert!(matches!(
        check_budget(usize::MAX / 4 + 1, 0, &limits),
        Err(DcStreamError::Limit("output byte count overflow"))
    ));
}

#[test]
fn dc_and_control_output_budgets_are_exact_before_allocation() {
    let jpeg = fixture(8, 8, &[(1, 1)], &[1]);
    let layout = TranscodeLayout::new(&jpeg).unwrap();
    let quant = quantization(&layout);
    let contexts = layout.dc_contexts();
    let geometry = geometry(&jpeg, &layout);
    let exact = PrepareLimits {
        max_samples: 4,
        max_output_bytes: 13,
    };
    prepare_dc_group(&layout, &quant, &contexts, geometry, 0, &exact).unwrap();
    for limits in [
        PrepareLimits {
            max_samples: 3,
            ..exact
        },
        PrepareLimits {
            max_output_bytes: 12,
            ..exact
        },
    ] {
        assert!(matches!(
            prepare_dc_group(&layout, &quant, &contexts, geometry, 0, &limits),
            Err(DcStreamError::Limit(_))
        ));
    }
    let maps = CflMaps {
        width: 1,
        height: 1,
        ytox: &[-128],
        ytob: &[127],
    };
    let exact = PrepareLimits {
        max_samples: 5,
        max_output_bytes: 20,
    };
    let prepared = prepare_ac_metadata_group(geometry, 0, maps, &exact).unwrap();
    assert_eq!(prepared.channels[0].samples, [-128]);
    assert_eq!(prepared.channels[1].samples, [127]);
    for limits in [
        PrepareLimits {
            max_samples: 4,
            ..exact
        },
        PrepareLimits {
            max_output_bytes: 19,
            ..exact
        },
    ] {
        assert!(matches!(
            prepare_ac_metadata_group(geometry, 0, maps, &limits),
            Err(DcStreamError::Limit(_))
        ));
    }
    assert!(
        check_budget(
            usize::MAX,
            1,
            &PrepareLimits {
                max_samples: usize::MAX,
                max_output_bytes: usize::MAX
            }
        )
        .is_err()
    );
    assert!(
        check_budget(
            usize::MAX / 4 + 1,
            0,
            &PrepareLimits {
                max_samples: usize::MAX,
                max_output_bytes: usize::MAX
            }
        )
        .is_err()
    );
    assert!(area(usize::MAX, 2).is_err());
}

#[test]
fn geometry_group_factor_and_map_mismatches_fail_without_partial_groups() {
    let jpeg = fixture(8, 8, &[(1, 1)], &[1]);
    let layout = TranscodeLayout::new(&jpeg).unwrap();
    let contexts = layout.dc_contexts();
    let geometry = geometry(&jpeg, &layout);
    let quant = quantization(&layout);
    let limits = PrepareLimits::default();
    assert!(prepare_dc_group(&layout, &quant, &contexts, geometry, usize::MAX, &limits).is_err());
    let different = FrameGeometry::new(17, 8, layout.subsampling()).unwrap();
    assert!(prepare_dc_group(&layout, &quant, &contexts, different, 0, &limits).is_err());
    // Equal padded dimensions must not hide a different channel sampling map.
    // Also cover gray raw 2x2 modes: effective shifts are zero, but modes differ.
    for sampling in [&[(2, 2), (1, 1), (1, 1)][..], &[(2, 2)][..]] {
        let jpeg = fixture(16, 16, sampling, &[1; 3][..sampling.len()]);
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let wrong_sampling =
            crate::jpeg::transcode_layout::ChromaSubsampling::from_sampling([1; 3], [1; 3])
                .unwrap();
        let wrong = FrameGeometry::new(16, 16, wrong_sampling).unwrap();
        let (width, height) = layout.dimensions_in_blocks();
        assert_eq!(
            wrong.dimensions_in_blocks(),
            (width as usize, height as usize)
        );
        assert_ne!(wrong.subsampling(), layout.subsampling());
        assert!(matches!(
            prepare_dc_group(
                &layout,
                &quantization(&layout),
                &layout.dc_contexts(),
                wrong,
                0,
                &limits
            ),
            Err(DcStreamError::Invalid(
                "frame subsampling disagrees with layout"
            ))
        ));
    }
    assert!(
        prepare_dc_group(
            &layout,
            &DcQuantization::default(),
            &contexts,
            geometry,
            0,
            &limits
        )
        .is_err()
    );
    for maps in [
        CflMaps {
            width: 0,
            height: 1,
            ytox: &[],
            ytob: &[],
        },
        CflMaps {
            width: 1,
            height: 1,
            ytox: &[],
            ytob: &[0],
        },
        CflMaps {
            width: usize::MAX,
            height: usize::MAX,
            ytox: &[],
            ytob: &[],
        },
    ] {
        assert!(prepare_ac_metadata_group(geometry, 0, maps, &limits).is_err());
    }
    let maps = CflMaps {
        width: 1,
        height: 1,
        ytox: &[0],
        ytob: &[0],
    };
    assert!(prepare_ac_metadata_group(geometry, usize::MAX, maps, &limits).is_err());
    assert!(
        channel_rect(
            BlockRect {
                x: 1,
                y: 0,
                width: 2,
                height: 2
            },
            1,
            1
        )
        .is_err()
    );
    assert!(
        channel_rect(
            BlockRect {
                x: 0,
                y: 0,
                width: 3,
                height: 2
            },
            1,
            1
        )
        .is_err()
    );
}
