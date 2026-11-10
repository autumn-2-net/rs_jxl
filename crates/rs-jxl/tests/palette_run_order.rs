// Bounded regressions for native run-based absolute palette ordering.
// Parent runs cargo tests centrally; no external decoder is required.

use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::modular::{
    group::{Channel, Transform},
    palette::{PaletteMode, PaletteOptions, apply_palette_range_with_options},
    predictor::Predictor,
    transforms::{
        self, PaletteGeometry, PaletteMapping, PaletteOrder, PaletteOutput, PaletteTools,
    },
};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode};

fn tools(order: PaletteOrder, mapping: PaletteMapping) -> PaletteTools {
    PaletteTools {
        order,
        mapping,
        ..PaletteTools::default()
    }
}

fn orders() -> [PaletteOrder; 7] {
    [
        PaletteOrder::RunFrequency,
        PaletteOrder::HotRuns { hot_colors: 2 },
        PaletteOrder::ScreenedRuns,
        PaletteOrder::AdjacencyMinLa { rounds: 0 },
        PaletteOrder::AdjacencyMinLa { rounds: 4 },
        PaletteOrder::EzengMinLa { rounds: 0 },
        PaletteOrder::EzengMinLa { rounds: 4 },
    ]
}

fn planes<const N: usize>(pixels: &[[i32; N]]) -> Vec<Vec<i32>> {
    (0..N)
        .map(|c| pixels.iter().map(|p| p[c]).collect())
        .collect()
}

fn forward(
    input: &[Vec<i32>],
    width: usize,
    depth: u32,
    ordered: bool,
    selected: PaletteTools,
) -> PaletteOutput {
    transforms::palette_forward_with_options(
        input,
        0,
        input.len(),
        width,
        input[0].len() / width,
        32,
        depth,
        ordered,
        selected,
    )
    .unwrap()
    .expect("fixture must activate palette")
}

fn tuples(output: &PaletteOutput) -> Vec<Vec<i32>> {
    (0..output.palette[0].len())
        .map(|i| output.palette.iter().map(|p| p[i]).collect())
        .collect()
}

// The two normative nonnegative implicit cubes, including low-depth aliases.
fn implicit(index: usize, channel: usize, depth: u32) -> i32 {
    if channel >= 3 {
        return 0;
    }
    let max = (1u64 << depth) - 1;
    if index < 64 {
        ((((index >> (2 * channel)) % 4) as u64 * max) >> 2) as i32
            + (1i32 << depth.saturating_sub(3))
    } else {
        (((((index - 64) / 5usize.pow(channel as u32)) % 5) as u64 * max) >> 2) as i32
    }
}

fn assert_expands(output: &PaletteOutput, input: &[Vec<i32>], depth: u32) {
    let size = output.palette[0].len();
    for (c, plane) in input.iter().enumerate() {
        let expanded: Vec<_> = output
            .indices
            .iter()
            .map(|&i| {
                let i = usize::try_from(i).unwrap();
                if i < size {
                    output.palette[c][i]
                } else {
                    implicit(i - size, c, depth)
                }
            })
            .collect();
        assert_eq!(&expanded, plane, "channel {c}");
    }
}

#[test]
fn source_modes_remain_zero_through_nine_only() {
    assert_eq!(PaletteOrder::default(), PaletteOrder::Default);
    assert_eq!(PaletteTools::default().order, PaletteOrder::Default);
    let source = [
        PaletteOrder::Default,
        PaletteOrder::FrequencyLuma,
        PaletteOrder::FrequencyBandsLuma,
        PaletteOrder::FrequencyBandsMorton,
        PaletteOrder::FrequentMorton,
        PaletteOrder::Morton,
        PaletteOrder::FrequencyBandsHilbert,
        PaletteOrder::Hilbert,
        PaletteOrder::Adjacency,
        PaletteOrder::FrequencyBandsAdjacency,
    ];
    for (i, expected) in source.into_iter().enumerate() {
        assert_eq!(PaletteOrder::from_source_mode(i as u8).unwrap(), expected);
    }
    for i in [10, 11, 255] {
        assert!(PaletteOrder::from_source_mode(i).is_err());
    }
}

#[test]
fn minla_zero_rounds_match_adjacency_and_large_tables_fall_back() {
    for colors in [7, 257] {
        let input: Vec<Vec<i32>> = (0..3)
            .map(|c| {
                (0..colors * 3)
                    .map(|i| ((i * 13 % colors) * 73 + c * 19 + 7) as i32)
                    .collect()
            })
            .collect();
        let run = |order| {
            transforms::palette_forward_with_options(
                &input,
                0,
                3,
                colors,
                3,
                512,
                16,
                true,
                tools(order, PaletteMapping::Flat),
            )
            .unwrap()
            .unwrap()
        };
        assert_eq!(
            run(PaletteOrder::AdjacencyMinLa { rounds: 0 }),
            run(PaletteOrder::Adjacency)
        );
        for order in [
            PaletteOrder::AdjacencyMinLa { rounds: 4 },
            PaletteOrder::EzengMinLa { rounds: 4 },
        ] {
            let result = run(order);
            assert_expands(&result, &input, 16);
            if colors > 256 {
                let fallback = match order {
                    PaletteOrder::AdjacencyMinLa { .. } => PaletteOrder::Adjacency,
                    _ => PaletteOrder::Default,
                };
                assert_eq!(result, run(fallback));
            }
        }
    }
}

#[test]
fn minla_round_limits_are_validated_even_without_palette_work() {
    for order in [
        PaletteOrder::AdjacencyMinLa { rounds: 41 },
        PaletteOrder::EzengMinLa { rounds: 255 },
    ] {
        assert!(
            PaletteOptions {
                tools: tools(order, PaletteMapping::Tree),
                ..PaletteOptions::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            transforms::palette_forward_with_options(
                &[vec![1, 2]],
                0,
                1,
                2,
                1,
                0,
                8,
                false,
                tools(order, PaletteMapping::Tree)
            )
            .is_err()
        );
    }
}

#[test]
fn minla_unsupported_channels_use_fallback_even_with_ignored_geometry() {
    let input = planes(&[[7, 13, 19, 0, -3], [29, 31, 37, 0, -7], [7, 13, 19, 0, -3]]);
    for (order, fallback) in [
        (
            PaletteOrder::AdjacencyMinLa { rounds: 4 },
            PaletteOrder::Adjacency,
        ),
        (
            PaletteOrder::EzengMinLa { rounds: 4 },
            PaletteOrder::Default,
        ),
    ] {
        for geometry in [
            PaletteGeometry::None,
            PaletteGeometry::Principal,
            PaletteGeometry::FrequencyBandsPrincipal,
        ] {
            let selected = PaletteTools {
                order,
                geometry,
                ..PaletteTools::default()
            };
            let output = forward(&input, 3, 16, true, selected);
            assert_eq!(
                output,
                forward(
                    &input,
                    3,
                    16,
                    true,
                    PaletteTools {
                        order: fallback,
                        ..selected
                    }
                )
            );
            assert_expands(&output, &input, 16);
        }
    }
}

#[test]
fn runs_not_pixels_and_hot_partitions_have_exact_orders() {
    let a = [10, 1, 2];
    let b = [90, 2, 3];
    let c = [30, 4, 5];
    // Pixel counts A=6, B=3, C=2; run counts A=1, B=3, C=2.
    let input = planes(&[a, a, a, a, a, a, b, c, b, c, b]);
    let cases = [
        (PaletteOrder::RunFrequency, vec![b, c, a]),
        (PaletteOrder::HotRuns { hot_colors: 0 }, vec![a, c, b]),
        (PaletteOrder::HotRuns { hot_colors: 1 }, vec![b, a, c]),
        (PaletteOrder::HotRuns { hot_colors: 2 }, vec![c, b, a]),
        (PaletteOrder::HotRuns { hot_colors: 3 }, vec![a, c, b]),
        (
            PaletteOrder::HotRuns {
                hot_colors: usize::MAX,
            },
            vec![a, c, b],
        ),
    ];
    for (order, expected) in cases {
        let mut previous = None;
        for mapping in [PaletteMapping::Tree, PaletteMapping::Flat] {
            let output = forward(&input, 11, 8, true, tools(order, mapping));
            assert_eq!(
                tuples(&output),
                expected.iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
                "{order:?}"
            );
            assert_expands(&output, &input, 8);
            if let Some(previous) = previous {
                assert_eq!(output, previous);
            }
            previous = Some(output);
        }
    }
}

#[test]
fn first_pixel_counts_and_runs_continue_across_rows() {
    let a = [10, 1, 2];
    let b = [90, 2, 3];
    for (pixels, width) in [
        (vec![b, a, b], 3),
        // A straddles the boundary: one A run, two B runs, not a tie.
        (vec![b, a, a, a, a, a, b, b], 4),
    ] {
        let input = planes(&pixels);
        for order in [
            PaletteOrder::RunFrequency,
            PaletteOrder::HotRuns { hot_colors: 1 },
        ] {
            let output = forward(&input, width, 8, true, tools(order, PaletteMapping::Tree));
            assert_eq!(tuples(&output), vec![b.to_vec(), a.to_vec()]);
            assert_eq!(
                output,
                forward(
                    &input,
                    pixels.len(),
                    8,
                    true,
                    tools(order, PaletteMapping::Tree)
                )
            );
            assert_expands(&output, &input, 8);
        }
    }
}

#[test]
fn full_signed_tuple_ties_include_channels_beyond_rgba() {
    let a = [7, 8, 9, 10, -20];
    let b = [7, 8, 9, 10, -2];
    let c = [7, 8, 9, -1, 99];
    // Identical RGB and full-tuple changes; all have two runs.
    let input = planes(&[b, a, c, b, c, a]);
    for order in [
        PaletteOrder::RunFrequency,
        PaletteOrder::HotRuns { hot_colors: 1 },
        PaletteOrder::HotRuns { hot_colors: 2 },
    ] {
        let tree = forward(&input, 3, 16, true, tools(order, PaletteMapping::Tree));
        assert_eq!(tuples(&tree), vec![c.to_vec(), a.to_vec(), b.to_vec()]);
        assert_eq!(
            tree,
            forward(&input, 3, 16, true, tools(order, PaletteMapping::Flat))
        );
        assert_expands(&tree, &input, 16);
    }
}

#[test]
fn screened_runs_preserves_full_tuples_at_small_and_sampled_sizes() {
    let fixtures = [
        // Hidden RGB, alpha-only changes, and the full unsigned 16-bit range.
        planes(&[
            [1001, 2003, 3007, 0],
            [4009, 5011, 6013, 0],
            [1001, 2003, 3007, 65535],
            [65535, 1, 32768, 0],
        ]),
        // Equal RGBA must not merge distinct signed fifth/sixth channels.
        planes(&[
            [7, 8, 9, 0, -32768, 32767],
            [7, 8, 9, 0, -2, 32767],
            [7, 8, 9, 0, -2, -32768],
            [-32768, 32767, -1, 0, -2, -32768],
        ]),
    ];
    for colors in fixtures {
        for repeats in [1, 1024] {
            let ids = [0, 0, 1, 0, 2, 1, 3, 3];
            let input: Vec<Vec<i32>> = colors
                .iter()
                .map(|p| (0..8 * repeats).map(|i| p[ids[i % 8]]).collect())
                .collect();
            for width in [1, 8, 8 * repeats] {
                let baseline = forward(&input, width, 16, true, PaletteTools::default());
                let mut expected = tuples(&baseline);
                expected.sort();
                assert_eq!(expected.len(), 4);
                let tree = forward(
                    &input,
                    width,
                    16,
                    true,
                    tools(PaletteOrder::ScreenedRuns, PaletteMapping::Tree),
                );
                let mut actual = tuples(&tree);
                actual.sort();
                assert_eq!(actual, expected, "preserve full tuple multiset");
                assert_expands(&tree, &input, 16);
                assert_eq!(
                    tree,
                    forward(
                        &input,
                        width,
                        16,
                        true,
                        tools(PaletteOrder::ScreenedRuns, PaletteMapping::Flat)
                    ),
                    "mapping must not change screening at width {width}"
                );
            }
        }
    }
}

#[test]
fn screened_runs_no_work_with_zero_or_one_explicit_candidate() {
    for pixels in [
        vec![[0, 0, 0], [255, 255, 255]],
        vec![[90, 1, 2], [0, 0, 0], [90, 1, 2]],
    ] {
        let input = planes(&pixels);
        for mapping in [PaletteMapping::Tree, PaletteMapping::Flat] {
            let baseline = forward(
                &input,
                pixels.len(),
                8,
                true,
                tools(PaletteOrder::Default, mapping),
            );
            assert!(baseline.palette[0].len() < 2);
            let output = forward(
                &input,
                pixels.len(),
                8,
                true,
                tools(PaletteOrder::ScreenedRuns, mapping),
            );
            assert_expands(&output, &input, 8);
            assert_eq!(output, baseline);
        }
    }
}

#[test]
fn implicit_transitions_are_counted_without_becoming_candidates() {
    let a = [90, 1, 2];
    let b = [10, 2, 3];
    let input = planes(&[a, [0; 3], a, b, b]);
    for order in [
        PaletteOrder::RunFrequency,
        PaletteOrder::HotRuns { hot_colors: 1 },
    ] {
        let output = forward(&input, 5, 8, true, tools(order, PaletteMapping::Flat));
        assert_eq!(tuples(&output), vec![a.to_vec(), b.to_vec()]);
        assert!(output.indices[1] >= 2);
        assert_expands(&output, &input, 8);
    }
}

#[test]
fn screened_runs_keeps_unpromoted_implicit_colors_out_of_explicit_table() {
    let a = [90, 1, 2];
    let b = [10, 2, 3];
    let input = planes(&[a, [0; 3], a, b, b, [255; 3], a, b]);
    for mapping in [PaletteMapping::Tree, PaletteMapping::Flat] {
        let output = forward(
            &input,
            4,
            8,
            true,
            tools(PaletteOrder::ScreenedRuns, mapping),
        );
        let mut table = tuples(&output);
        table.sort();
        assert_eq!(table, vec![b.to_vec(), a.to_vec()]);
        for pos in [1, 5] {
            assert!(output.indices[pos] >= 2, "implicit colors stay implicit");
        }
        assert_expands(&output, &input, 8);
    }
}

#[test]
fn implicit_only_aliases_and_duplicate_promotions_keep_exact_indices() {
    for depth in [1, 2, 8] {
        let input: Vec<Vec<i32>> = (0..3)
            .map(|c| (0..189).map(|i| implicit(i, c, depth)).collect())
            .collect();
        let baseline = forward(&input, 21, depth, false, PaletteTools::default());
        for order in orders() {
            let tree = forward(&input, 21, depth, true, tools(order, PaletteMapping::Tree));
            let flat = forward(&input, 21, depth, true, tools(order, PaletteMapping::Flat));
            assert_eq!(tree, flat);
            assert_eq!(
                tree.palette[0].len(),
                baseline.palette[0].len(),
                "preserve duplicate promotions at depth {depth}"
            );
            assert_expands(&tree, &input, depth);
            let table = tuples(&tree);
            let mut actual_colors = table.clone();
            let mut original_colors = tuples(&baseline);
            actual_colors.sort();
            original_colors.sort();
            assert_eq!(
                actual_colors, original_colors,
                "preserve every promoted alias"
            );
            for (pos, &index) in tree.indices.iter().enumerate() {
                let color: Vec<_> = input.iter().map(|p| p[pos]).collect();
                let expected = table.iter().rposition(|p| *p == color).unwrap_or_else(|| {
                    table.len()
                        + (0..189)
                            .rfind(|&i| (0..3).all(|c| implicit(i, c, depth) == color[c]))
                            .unwrap()
                });
                assert_eq!(index as usize, expected, "last alias must win");
            }
            if depth == 8 {
                assert!(tree.palette[0].is_empty());
                assert_eq!(tree, baseline);
            }
        }
    }
    let b = [10, 2, 3];
    for count in [10, 11] {
        let mut pixels = vec![[0; 3]; count];
        pixels.extend([b, [90, 1, 2], b]);
        let input = planes(&pixels);
        for order in [
            PaletteOrder::RunFrequency,
            PaletteOrder::HotRuns { hot_colors: 1 },
        ] {
            let output = forward(
                &input,
                pixels.len(),
                8,
                true,
                tools(order, PaletteMapping::Flat),
            );
            assert_eq!(output.palette[0].len(), if count == 10 { 2 } else { 3 });
            assert_eq!(
                tuples(&output)[0],
                b.to_vec(),
                "promoted black has more pixels but only one run"
            );
            assert_expands(&output, &input, 8);
        }
    }
}

#[test]
fn unordered_geometry_and_fewer_than_three_channels_bypass_new_orders() {
    let rgb = planes(&[[90, 1, 2], [10, 2, 3], [90, 1, 2], [30, 4, 5]]);
    for order in orders() {
        for mapping in [PaletteMapping::Tree, PaletteMapping::Flat] {
            let selected = tools(order, mapping);
            assert_eq!(
                forward(&rgb, 2, 8, false, selected),
                forward(&rgb, 2, 8, false, tools(PaletteOrder::Default, mapping))
            );
            for geometry in [
                PaletteGeometry::Principal,
                PaletteGeometry::FrequencyBandsPrincipal,
            ] {
                assert_eq!(
                    forward(
                        &rgb,
                        2,
                        8,
                        true,
                        PaletteTools {
                            geometry,
                            ..selected
                        }
                    ),
                    forward(
                        &rgb,
                        2,
                        8,
                        true,
                        PaletteTools {
                            geometry,
                            order: PaletteOrder::Default,
                            mapping
                        }
                    ),
                );
            }
            for count in [1, 2] {
                let input = rgb[..count].to_vec();
                assert_eq!(
                    forward(&input, 2, 8, true, selected),
                    forward(&input, 2, 8, true, tools(PaletteOrder::Default, mapping))
                );
            }
        }
    }
}

fn apply(
    input: &[Vec<i32>],
    width: usize,
    depth: u8,
    limit: u32,
    mode: PaletteMode,
    selected: PaletteTools,
) -> (Vec<Channel>, Transform) {
    let mut channels: Vec<_> = input
        .iter()
        .map(|samples| Channel {
            width,
            height: samples.len() / width,
            hshift: 0,
            vshift: 0,
            samples: samples.clone(),
        })
        .collect();
    let mut meta = 0;
    let transform = apply_palette_range_with_options(
        &mut channels,
        &mut meta,
        depth,
        0,
        input.len(),
        limit,
        true,
        mode,
        selected,
    )
    .unwrap()
    .expect("activate palette policy");
    assert_eq!(meta, 1);
    (channels, transform)
}

#[test]
fn successful_delta_ignores_order_fallback_orders_and_zero_prefix_is_added_last() {
    let success = planes(&[[20, 40, 60], [30, 60, 90], [50, 100, 150], [70, 140, 210]]);
    let residual = transforms::delta_palette_forward(&success, 0, 3, 4, 1, 3, 8, Predictor::Left)
        .unwrap()
        .unwrap();
    assert_eq!(residual.num_deltas, 2);
    assert_eq!(
        residual.palette,
        vec![vec![20, 10], vec![40, 20], vec![60, 30]]
    );
    let baseline = apply(
        &success,
        4,
        8,
        3,
        PaletteMode::ExactDelta(Predictor::Left),
        PaletteTools::default(),
    );
    let fallback = planes(&[
        [10, 20, 30],
        [10, 20, 30],
        [50, 60, 70],
        [10, 20, 30],
        [110, 140, 170],
        [50, 60, 70],
        [110, 140, 170],
        [110, 140, 170],
    ]);
    assert!(
        transforms::delta_palette_forward(&fallback, 0, 3, 8, 1, 3, 8, Predictor::Left)
            .unwrap()
            .is_none()
    );
    for order in orders() {
        let selected = tools(order, PaletteMapping::Flat);
        assert_eq!(
            apply(
                &success,
                4,
                8,
                3,
                PaletteMode::ExactDelta(Predictor::Left),
                selected
            ),
            baseline
        );
        assert_eq!(
            apply(
                &fallback,
                8,
                8,
                3,
                PaletteMode::ExactDelta(Predictor::Left),
                selected
            ),
            apply(&fallback, 8, 8, 3, PaletteMode::Absolute, selected)
        );
        let absolute = forward(&fallback, 8, 8, true, selected);
        assert_expands(&absolute, &fallback, 8);
        let mut actual_colors = tuples(&absolute);
        actual_colors.sort();
        let mut expected_colors = tuples(&forward(&fallback, 8, 8, true, PaletteTools::default()));
        expected_colors.sort();
        assert_eq!(
            actual_colors, expected_colors,
            "fallback is a valid permutation"
        );
        let mask = transforms::exact_zero_delta_mask(&fallback, 0, 3, 8, 1, 8, Predictor::Left)
            .unwrap()
            .unwrap();
        assert!(mask.iter().any(|&v| v != 0));
        let expected = transforms::add_exact_zero_delta(&absolute, Predictor::Left, &mask).unwrap();
        let (actual, transform) = apply(
            &fallback,
            8,
            8,
            3,
            PaletteMode::ExactZero(Predictor::Left),
            selected,
        );
        assert!(matches!(transform, Transform::Palette { deltas: 1, .. }));
        assert_eq!(
            actual[0].samples,
            expected.palette.into_iter().flatten().collect::<Vec<_>>()
        );
        assert_eq!(actual[1].samples, expected.indices);
    }
}

#[test]
fn complete_rgb_and_rgba16_streams_preserve_hidden_samples_and_delta_modes() {
    let colors = [
        [1001u16, 2003, 3007, 0],
        [4009, 5011, 6013, 0],
        [1001, 2003, 3007, 65535],
    ];
    let ids = [0, 0, 1, 0, 2, 1, 2, 2];
    for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
        let count = if layout == PixelLayout::Rgb { 3 } else { 4 };
        let samples: Vec<_> = (0..32)
            .flat_map(|i| colors[ids[i % 8]][..count].to_vec())
            .collect();
        let pixels = if count == 3 {
            PixelData::U8(samples.iter().map(|&v| (v / 257) as u8).collect())
        } else {
            PixelData::U16(samples)
        };
        let image = Image::new(8, 4, layout, pixels).unwrap();
        let input = match image.pixels() {
            PixelData::U8(p) => (0..count)
                .map(|c| p.chunks_exact(count).map(|v| i32::from(v[c])).collect())
                .collect::<Vec<_>>(),
            PixelData::U16(p) => (0..count)
                .map(|c| p.chunks_exact(count).map(|v| i32::from(v[c])).collect())
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };
        for order in orders() {
            for (mapping, mode) in [
                (PaletteMapping::Tree, PaletteMode::Absolute),
                (PaletteMapping::Flat, PaletteMode::Absolute),
                (
                    PaletteMapping::Flat,
                    PaletteMode::ExactDelta(Predictor::Left),
                ),
                (
                    PaletteMapping::Tree,
                    PaletteMode::ExactZero(Predictor::Left),
                ),
            ] {
                let selected = tools(order, mapping);
                let (_, transform) = apply(&input, 8, image.bits(), 3, mode, selected);
                assert!(
                    matches!(transform, Transform::Palette { channels, .. } if channels == count)
                );
                let options = EncodeOptions {
                    modular: ModularOptions {
                        rct: None,
                        palette_policy: Some(PaletteOptions {
                            palette_colors: 3,
                            mode,
                            tools: selected,
                            channel_colors_pre_transform_percent: 0.0,
                            channel_colors_percent: 0.0,
                            ..PaletteOptions::default()
                        }),
                        ..ModularOptions::default()
                    },
                    ..EncodeOptions::default()
                };
                let bytes = encode(&image, &options).unwrap();
                let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
                assert!(decoded.complete);
                assert_eq!(decoded.frames.len(), 1);
                assert_eq!(decoded.info.size, (8, 4));
                assert_eq!(
                    &decoded.frames[0].pixels,
                    image.pixels(),
                    "{layout:?} {order:?} {mapping:?} {mode:?}"
                );
            }
        }
    }
}
