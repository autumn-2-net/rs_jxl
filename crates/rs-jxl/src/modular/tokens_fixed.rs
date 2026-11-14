// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Fixed-tree emission from context_predict.h/encoding.h: compute only the
//! property that selects a context, without constructing the generic MA state.

use super::{Channel, ChannelTree, Literal, Predictor, Result, pack_signed, weighted};
use crate::modular::predictor::clamped_gradient;

pub(super) fn emit(
    plane: &Channel,
    plan: &ChannelTree,
    header: &weighted::Header,
    history: Option<&[[i32; 2]]>,
    mut emit: impl FnMut(Literal),
) -> Result<bool> {
    let Some(contexts) = plan.simple_contexts() else {
        return Ok(false);
    };
    let width = plane.width;
    if width == 0 || width.checked_mul(plane.height) != Some(plane.samples.len()) {
        return Err(crate::Error::Encoding(
            "invalid fixed-token plane dimensions".into(),
        ));
    }
    if contexts.predictor == Predictor::Gradient {
        for (y, current) in plane.samples.chunks_exact(width).enumerate() {
            if y == 0 {
                // All prediction neighbors collapse to W on the first row.
                let mut left = 0;
                for &value in current {
                    emit(Literal {
                        context: contexts.lookup(i64::from(left).clamp(-8192, 8191)),
                        value: pack_signed(value.wrapping_sub(left)),
                    });
                    left = value;
                }
                continue;
            }
            let above = &plane.samples[(y - 1) * width..y * width];
            // At x=0, W=N=NW; routing and prediction both use N.
            emit(Literal {
                context: contexts.lookup(i64::from(above[0]).clamp(-8192, 8191)),
                value: pack_signed(current[0].wrapping_sub(above[0])),
            });
            for (((&value, &left), &top), &northwest) in current[1..]
                .iter()
                .zip(&current[..width - 1])
                .zip(&above[1..])
                .zip(&above[..width - 1])
            {
                // Gradient-LUT routing clamps a wide sum, unlike generic i32 properties.
                let gradient =
                    (i64::from(left) + i64::from(top) - i64::from(northwest)).clamp(-8192, 8191);
                let residual = value.wrapping_sub(clamped_gradient(left, top, northwest));
                emit(Literal {
                    context: contexts.lookup(gradient),
                    value: pack_signed(residual),
                });
            }
        }
    } else if let Some(history) = history {
        if history.len() != plane.samples.len() {
            return Err(crate::Error::Encoding(
                "Weighted history length mismatch".into(),
            ));
        }
        for (&value, &[prediction, error]) in plane.samples.iter().zip(history) {
            emit(Literal {
                context: contexts.lookup(i64::from(error)),
                value: pack_signed(value.wrapping_sub(prediction)),
            });
        }
    } else {
        emit_weighted(plane, header, |(residual, error)| {
            emit(Literal {
                context: contexts.lookup(i64::from(error)),
                value: pack_signed(residual),
            });
        })?;
    }
    Ok(true)
}

// Only five neighbors are needed, and only the two edge pixels need boundary
// handling. Keep the wide arithmetic: the optional narrow state kernel adds
// per-pixel range checks and has not established a default speed advantage.
fn emit_weighted(
    plane: &Channel,
    header: &weighted::Header,
    mut emit: impl FnMut((i32, i32)),
) -> Result<()> {
    let width = plane.width;
    let mut state = weighted::State::new(*header, width)?;
    for (y, current) in plane.samples.chunks_exact(width).enumerate() {
        if y == 0 {
            let mut left = 0;
            for (x, &value) in current.iter().enumerate() {
                // Samples collapse to W, but weighted error history must still
                // advance even on the first row or a constant/alpha plane.
                emit(weighted_pixel::<false>(&mut state, value, x, y, [left; 5]));
                left = value;
            }
            continue;
        }
        let top = &plane.samples[(y - 1) * width..y * width];
        let upper = if y > 1 {
            &plane.samples[(y - 2) * width..(y - 1) * width]
        } else {
            top
        };
        emit(weighted_pixel::<false>(
            &mut state,
            current[0],
            0,
            y,
            [
                top[0],
                top[0],
                if width > 1 { top[1] } else { top[0] },
                top[0],
                upper[0],
            ],
        ));
        // The validated plane and this range prove 0 < x < width - 1.
        for x in 1..width - 1 {
            emit(weighted_pixel::<true>(
                &mut state,
                current[x],
                x,
                y,
                [top[x], current[x - 1], top[x + 1], top[x - 1], upper[x]],
            ));
        }
        if width > 1 {
            let x = width - 1;
            emit(weighted_pixel::<false>(
                &mut state,
                current[x],
                x,
                y,
                [top[x], current[x - 1], top[x], top[x - 1], upper[x]],
            ));
        }
    }
    Ok(())
}

#[inline]
fn weighted_pixel<const INTERIOR: bool>(
    state: &mut weighted::State,
    value: i32,
    x: usize,
    y: usize,
    [n, w, ne, nw, nn]: [i32; 5],
) -> (i32, i32) {
    let (prediction, error) = if INTERIOR {
        state.predict_and_update_raster_interior::<false>(value, x, y, n, w, ne, nw, nn)
    } else {
        state.predict_and_update_raster::<false>(value, x, y, n, w, ne, nw, nn)
    };
    (value.wrapping_sub(prediction), error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::tree::Tree;
    use crate::modular::{encode, fixed, tree::Node};

    fn compare(plane: &Channel, tree: &Tree, header: &weighted::Header) {
        compare_channel(std::slice::from_ref(plane), 0, 0, tree, header);
    }

    fn compare_channel(
        planes: &[Channel],
        channel: usize,
        group: usize,
        tree: &Tree,
        header: &weighted::Header,
    ) {
        let plan = ChannelTree::new(tree, channel, group);
        let mut actual = Vec::new();
        assert!(emit(&planes[channel], &plan, header, None, |t| actual.push(t)).unwrap());
        let mut expected = Vec::new();
        encode::visit_channel(
            planes,
            channel,
            header,
            plan.weighted,
            plan.previous_channels,
            group,
            |_, _, neighbors, weighted, properties, value| {
                let leaf = plan.lookup_pixel(properties, neighbors)?;
                let prediction = crate::modular::predictor::predict_one(
                    leaf.predictor,
                    neighbors,
                    Some(i64::from(weighted)),
                )
                .unwrap();
                expected.push(Literal {
                    context: leaf.context as u32,
                    value: pack_signed((i64::from(value) - prediction) as i32),
                });
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(actual.len(), planes[channel].samples.len());
        assert_eq!(actual, expected);
    }

    #[test]
    fn weighted_peeled_rows_match_generic_for_presets_and_full_i32_patterns() {
        let tree = fixed::build(
            31,
            fixed::FixedTreeMode::Weighted {
                total_pixels: 65536,
            },
        )
        .unwrap();
        let mut headers: Vec<_> = (0..5)
            .map(|mode| weighted::Header::preset(mode).unwrap())
            .collect();
        for w in [[0; 4], [15; 4]] {
            headers.push(weighted::Header {
                p1_c: 31,
                p2_c: 31,
                p3_ca: 31,
                p3_cb: 31,
                p3_cc: 31,
                p3_cd: 31,
                p3_ce: 31,
                w,
            });
        }
        for width in [1, 2, 3, 8, 17, 33] {
            for height in [0, 1, 2, 3, 9] {
                for pattern in 0..5 {
                    let mut random = 0x58c3_71b9u32;
                    let plane = Channel {
                        width,
                        height,
                        hshift: 0,
                        vshift: 0,
                        samples: (0..width * height)
                            .map(|i| {
                                random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                                match pattern {
                                    0 => 0,
                                    1 => i32::MIN,
                                    2 => i32::MAX,
                                    3 => [128_000_000, -128_000_000, 0][i % 3],
                                    _ => random as i32,
                                }
                            })
                            .collect(),
                    };
                    for header in &headers {
                        compare(&plane, &tree, header);
                    }
                }
            }
        }
    }

    #[test]
    fn static_channel_group_routing_preserves_contexts_and_hidden_rgb() {
        let weighted = Node::Leaf {
            predictor: Predictor::Weighted,
            offset: 0,
            multiplier: 1,
        };
        let gradient = Node::Leaf {
            predictor: Predictor::Gradient,
            offset: 0,
            multiplier: 1,
        };
        let tree = Tree::new(vec![
            Node::Split {
                property: 0,
                threshold: 0,
                greater: 2,
                less_equal: 1,
            },
            weighted.clone(),
            Node::Split {
                property: 1,
                threshold: 3,
                greater: 3,
                less_equal: 6,
            },
            Node::Split {
                property: 15,
                threshold: 0,
                greater: 4,
                less_equal: 5,
            },
            weighted.clone(),
            weighted,
            Node::Split {
                property: 9,
                threshold: 0,
                greater: 7,
                less_equal: 8,
            },
            gradient.clone(),
            gradient,
        ])
        .unwrap();
        for width in [1, 2, 3, 8, 33] {
            // Modular's alpha-first ordering: zero alpha must not suppress RGB.
            let planes: Vec<_> = (0..4)
                .map(|c| Channel {
                    width,
                    height: 4,
                    hshift: 0,
                    vshift: 0,
                    samples: (0..width * 4)
                        .map(|i| {
                            if c == 0 {
                                0
                            } else {
                                [i32::MIN, i32::MAX, -8193, 8192, 65535, 0, -1][(i + c) % 7]
                            }
                        })
                        .collect(),
                })
                .collect();
            for group in [0, 4] {
                for c in 0..planes.len() {
                    compare_channel(&planes, c, group, &tree, &weighted::Header::default());
                }
            }
        }
    }

    #[test]
    fn fixed_context_map_matches_generic_emission_at_borders_and_wide_values() {
        for width in [1, 2, 7, 256, 257] {
            for bits in [8, 16, 31] {
                let plane = Channel {
                    width,
                    height: 9,
                    hshift: 0,
                    vshift: 0,
                    samples: (0..width * 9)
                        .map(|i| {
                            if bits == 31 {
                                [i32::MIN, i32::MAX, -8193, 8192, -1, 0, 1][i % 7]
                            } else {
                                ((i * 997 + i / 3 * 47) % (1 << bits)) as i32
                            }
                        })
                        .collect(),
                };
                for population in [1, 1024, 5000, 65536] {
                    for mode in [
                        fixed::FixedTreeMode::Gradient {
                            total_pixels: population,
                            previous_channels: 0,
                        },
                        fixed::FixedTreeMode::Weighted {
                            total_pixels: population,
                        },
                    ] {
                        let tree = fixed::build(bits, mode).unwrap();
                        compare(&plane, &tree, &weighted::Header::default());
                    }
                }
            }
        }
    }

    #[test]
    fn general_or_offset_trees_keep_the_generic_path() {
        let plane = Channel {
            width: 2,
            height: 1,
            hshift: 0,
            vshift: 0,
            samples: vec![1, 2],
        };
        let trees = [
            fixed::build(
                8,
                fixed::FixedTreeMode::Gradient {
                    total_pixels: 65536,
                    previous_channels: 1,
                },
            )
            .unwrap(),
            Tree::new(vec![Node::Leaf {
                predictor: Predictor::Gradient,
                offset: 1,
                multiplier: 1,
            }])
            .unwrap(),
            Tree::new(vec![Node::Leaf {
                predictor: Predictor::Weighted,
                offset: 0,
                multiplier: 2,
            }])
            .unwrap(),
        ];
        for tree in trees {
            assert!(
                !emit(
                    &plane,
                    &ChannelTree::new(&tree, 0, 0),
                    &weighted::Header::default(),
                    None,
                    |_| panic!("ineligible tree emitted a token")
                )
                .unwrap()
            );
        }
    }

    #[test]
    fn clipped_gradient_never_enters_unreachable_boundary_leaf() {
        let plane = Channel {
            width: 3,
            height: 2,
            hshift: 0,
            vshift: 0,
            samples: vec![i32::MIN, i32::MIN, i32::MAX, i32::MIN, i32::MAX, 0],
        };
        let leaf = Node::Leaf {
            predictor: Predictor::Gradient,
            offset: 0,
            multiplier: 1,
        };
        for threshold in [-8193, -8192, 8190] {
            let tree = Tree::new(vec![
                Node::Split {
                    property: 9,
                    threshold,
                    greater: 1,
                    less_equal: 2,
                },
                leaf.clone(),
                leaf.clone(),
            ])
            .unwrap();
            compare(&plane, &tree, &weighted::Header::default());
        }
    }

    #[test]
    fn gradient_peeled_edges_match_generic_for_constant_extreme_planes() {
        let leaf = Node::Leaf {
            predictor: Predictor::Gradient,
            offset: 0,
            multiplier: 1,
        };
        for threshold in [-8193, -8192, -1, 0, 8190] {
            let tree = Tree::new(vec![
                Node::Split {
                    property: 9,
                    threshold,
                    greater: 1,
                    less_equal: 2,
                },
                leaf.clone(),
                leaf.clone(),
            ])
            .unwrap();
            for width in [1, 2, 3, 8, 17] {
                for height in [0, 1, 2, 3] {
                    for value in [i32::MIN, -8193, -8192, -1, 0, 1, 8191, 8192, i32::MAX] {
                        let plane = Channel {
                            width,
                            height,
                            hshift: 0,
                            vshift: 0,
                            samples: vec![value; width * height],
                        };
                        compare(&plane, &tree, &weighted::Header::default());
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_plane_returns_error_before_emission() {
        let tree = fixed::build(
            8,
            fixed::FixedTreeMode::Weighted {
                total_pixels: 65536,
            },
        )
        .unwrap();
        for (width, height, count) in [(0, 1, 1), (2, 2, 3), (usize::MAX, 2, 1)] {
            let plane = Channel {
                width,
                height,
                hshift: 0,
                vshift: 0,
                samples: vec![0; count],
            };
            assert!(
                emit(
                    &plane,
                    &ChannelTree::new(&tree, 0, 0),
                    &weighted::Header::default(),
                    None,
                    |_| panic!("invalid plane emitted a token")
                )
                .is_err()
            );
        }
    }
}
