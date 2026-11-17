// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Restricted experimental access to Modular preparation and causal inversion.
//! Source: enc_modular.cc/context_predict.h and rct.cc::InvRCTRow at
//! b866b3defec13494bfbaf705db93581cda118e37. No standard encoder path changes.
use super::{
    encode::{self, ModularOptions},
    group::{Channel, Transform},
    predictor::{Neighbors, predict_one},
    tree::{ChannelTree, Properties, Tree},
    weighted,
};
use crate::{Error, Image, Result, entropy::stream::Token};
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub stream_id: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneLayout {
    pub width: usize,
    pub height: usize,
    pub hshift: i32,
    pub vshift: i32,
}
pub struct PreparedGroup {
    pub planes: Vec<Channel>,
    pub tree: Tree,
    pub wp: weighted::Header,
    pub transforms: Vec<Transform>,
    pub tokens: Vec<Token>,
    pub stream_id: usize,
    pub previous_channels: usize,
}
impl PreparedGroup {
    pub fn layout(&self) -> Vec<PlaneLayout> {
        self.planes
            .iter()
            .map(|p| PlaneLayout {
                width: p.width,
                height: p.height,
                hshift: p.hshift,
                vshift: p.vshift,
            })
            .collect()
    }
}
fn invalid(s: &str) -> Error {
    Error::InvalidInput(s.into())
}
fn plane_samples(width: usize, height: usize, hshift: i32, vshift: i32) -> Result<usize> {
    // Properties store group-local positions as i32. The shift domain matches
    // modular/layout.rs::validate_shapes and metadata's dim_shift field.
    if width == 0
        || height == 0
        || width > i32::MAX as usize
        || height > i32::MAX as usize
        || !(0..=31).contains(&hshift)
        || !(0..=31).contains(&vshift)
    {
        return Err(invalid("Plus bridge plane layout"));
    }
    width
        .checked_mul(height)
        .ok_or(Error::Limit("Plus bridge samples"))
}
/// Prepare the actual Modular prediction domain. This restricted profile has
/// no LZ, palette, squeeze, patches or refinement. RCT preserves every channel,
/// including hidden RGB; extra channels are neither dropped nor premultiplied.
pub fn prepare(image: &Image, rect: Rect, options: &ModularOptions) -> Result<PreparedGroup> {
    let resolved = options.for_pixel_metadata(image.metadata());
    let options = resolved.as_ref();
    options.validate()?;
    if options.lz77
        || options.palette_colors.is_some()
        || !options.squeeze.is_empty()
        || options.rct_candidates.is_some()
        || options.refinement.enabled()
    {
        return Err(invalid("unsupported Plus bridge tool configuration"));
    }
    if rect.width == 0
        || rect.height == 0
        || rect.width > i32::MAX as usize
        || rect.height > i32::MAX as usize
        || rect.stream_id > i32::MAX as usize
        || rect
            .x
            .checked_add(rect.width)
            .is_none_or(|v| v > image.width() as usize)
        || rect
            .y
            .checked_add(rect.height)
            .is_none_or(|v| v > image.height() as usize)
    {
        return Err(invalid("Plus bridge rectangle outside image"));
    }
    let p = encode::prepare(
        image,
        encode::Group {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            stream_id: rect.stream_id,
        },
        options,
    )?;
    let tree = encode::learn(&p.planes, options, &p.wp, p.stream_id, image.bits(), None)?;
    let (tokens, has_matches) =
        encode::group_stream(&p.planes, &tree, options, &p.wp, p.stream_id)?;
    if has_matches || tokens.iter().any(|t| t.is_length) {
        return Err(invalid("Plus bridge unexpected LZ token"));
    }
    Ok(PreparedGroup {
        planes: p.planes,
        tree,
        wp: p.wp,
        transforms: p.transforms,
        tokens,
        stream_id: p.stream_id,
        previous_channels: options.previous_channels,
    })
}

/// Parse a bounded stream of residual literals using the existing Modular LZ
/// tool. Callers may filter zero residuals before this stage. The input must
/// contain literal tokens only, with contexts in the original MA leaf domain.
/// This experimental bridge caps the input at 2^24 literals and 2^20 leaves;
/// it does not alter `prepare` or enable LZ on the standard encoder path.
pub fn parse_residual_tokens(
    raw: &[Token],
    leaves: usize,
    distance_multiplier: u32,
    options: &ModularOptions,
) -> Result<(Vec<Token>, bool)> {
    options.validate()?;
    if leaves == 0 || leaves > 1 << 20 || raw.len() > 1 << 24 {
        return Err(invalid("Plus bridge residual token limits"));
    }
    if raw.iter().any(|t| t.is_length || t.context >= leaves) {
        return Err(invalid("Plus bridge requires literal MA contexts"));
    }
    let mut literals = Vec::new();
    literals
        .try_reserve_exact(raw.len())
        .map_err(|_| Error::Limit("Plus bridge literal allocation"))?;
    literals.extend(raw.iter().map(|t| crate::tools::lz77::Literal {
        context: t.context as u32,
        value: t.value,
    }));
    encode::parse_group_tokens(&literals, leaves, distance_multiplier as usize, options)
}
/// Decode only the transformed planes from bounded, signaled geometry and a
/// decoded MA tree/WP header. The callback supplies packed signed residuals,
/// not original samples or a saved context trace. All output is local until
/// success. max_samples bounds the sum across channels before allocation.
pub fn decode_planes(
    layout: &[PlaneLayout],
    tree: &Tree,
    wp: &weighted::Header,
    stream_id: usize,
    previous_channels: usize,
    max_samples: usize,
    mut read_residual: impl FnMut(usize) -> Result<u32>,
) -> Result<Vec<Channel>> {
    wp.validate()?;
    if layout.is_empty()
        || layout.len() > 256
        || previous_channels > 60
        || stream_id > i32::MAX as usize
    {
        return Err(invalid("Plus bridge decode configuration"));
    }
    let mut total = 0usize;
    for p in layout {
        total = total
            .checked_add(plane_samples(p.width, p.height, p.hshift, p.vshift)?)
            .ok_or(Error::Limit("Plus bridge samples"))?;
    }
    if total > max_samples {
        return Err(Error::Limit("Plus bridge samples"));
    }
    let mut planes: Vec<Channel> = Vec::with_capacity(layout.len());

    for (c, p) in layout.iter().enumerate() {
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(p.width * p.height)
            .map_err(|_| Error::Limit("Plus bridge allocation"))?;
        samples.resize(p.width * p.height, 0);
        let plan = ChannelTree::new(tree, c, stream_id);
        let fixed_leaf = plan.leaf();
        let references: Vec<_> = planes
            .iter()
            .rev()
            .filter(|r| {
                r.width == p.width
                    && r.height == p.height
                    && r.hshift == p.hshift
                    && r.vshift == p.vshift
            })
            .take(if fixed_leaf.is_some() {
                0
            } else {
                previous_channels
            })
            .collect();
        let mut props = if fixed_leaf.is_some() {
            Vec::new()
        } else {
            vec![0; 16 + 4 * previous_channels]
        };
        let mut properties = Properties::default();
        let mut state = if plan.weighted {
            Some(weighted::State::new(*wp, p.width)?)
        } else {
            None
        };
        for y in 0..p.height {
            for x in 0..p.width {
                let pos = y * p.width + x;
                let n = Neighbors::from_plane(&samples, p.width, p.width, x, y)
                    .map_err(|e| invalid(&e.to_string()))?;
                let (prediction, error) = if let Some(state) = &mut state {
                    state.predict(x, y, n.top, n.left, n.topright, n.topleft, n.toptop)?
                } else {
                    (0, 0)
                };
                if fixed_leaf.is_none() {
                    props[..16]
                        .copy_from_slice(&properties.calculate(n, x, y, c, stream_id, error));
                }
                for (i, r) in references.iter().enumerate() {
                    let value = r.samples[pos];
                    let left = if x > 0 { r.samples[pos - 1] } else { 0 };
                    let top = if y > 0 {
                        r.samples[pos - p.width]
                    } else {
                        left
                    };
                    let nw = if x > 0 && y > 0 {
                        r.samples[pos - p.width - 1]
                    } else {
                        left
                    };
                    let residual =
                        value.wrapping_sub(super::predictor::clamped_gradient(left, top, nw));
                    props[16 + 4 * i..20 + 4 * i].copy_from_slice(&[
                        value.wrapping_abs(),
                        value,
                        residual.wrapping_abs(),
                        residual,
                    ]);
                }
                // ChannelTree has already specialized channel/group properties.
                // A remaining single leaf needs no properties/history from
                // other channels, but still uses identical causal prediction.
                let leaf = if let Some(leaf) = fixed_leaf {
                    leaf
                } else {
                    plan.lookup_pixel(&props, n)?
                };
                let (context, predictor, offset, multiplier) =
                    (leaf.context, leaf.predictor, leaf.offset, leaf.multiplier);
                let packed = read_residual(context)?;
                let residual = ((packed >> 1) as i32) ^ -((packed & 1) as i32);
                let guess = predict_one(predictor, n, Some(i64::from(prediction)))
                    .map_err(|e| invalid(&e.to_string()))?;
                // Source Modular PixelAdd/multiplier arithmetic is modulo 2^32.
                // i64 can hold each predictor and signed residual product; the
                // additions below intentionally wrap before narrowing to i32.
                let value = guess
                    .wrapping_add(i64::from(offset))
                    .wrapping_add(i64::from(residual) * i64::from(multiplier))
                    as i32;
                samples[pos] = value;
                if let Some(state) = &mut state {
                    state.update(value, x, y)?;
                }
            }
        }
        planes.push(Channel {
            width: p.width,
            height: p.height,
            hshift: p.hshift,
            vshift: p.vshift,
            samples,
        });
    }
    Ok(planes)
}
/// Invert only the RCT transform subset emitted by this profile. Other tools
/// must acquire independent inverse coverage before they can be admitted.
pub fn inverse_transforms(planes: &mut [Channel], transforms: &[Transform]) -> Result<()> {
    for p in planes.iter() {
        if plane_samples(p.width, p.height, p.hshift, p.vshift)? != p.samples.len() {
            return Err(invalid("Plus bridge inverse sample count"));
        }
    }
    for t in transforms {
        let (begin, kind) = match t {
            Transform::Rct(k) => (0, *k),
            Transform::RctAt { begin, kind } => (*begin, *kind),
            _ => return Err(invalid("Plus bridge unsupported inverse transform")),
        };
        let end = begin
            .checked_add(3)
            .ok_or_else(|| invalid("Plus bridge RCT range"))?;
        if kind >= 42 || end > planes.len() {
            return Err(invalid("Plus bridge RCT range"));
        }
        let first = &planes[begin];
        if planes[begin..end].iter().any(|p| {
            p.width != first.width
                || p.height != first.height
                || p.hshift != first.hshift
                || p.vshift != first.vshift
                || p.samples.len() != first.samples.len()
        }) {
            return Err(invalid("Plus bridge RCT geometry"));
        }
    }
    for t in transforms.iter().rev() {
        let (begin, kind) = match t {
            Transform::Rct(k) => (0, *k),
            Transform::RctAt { begin, kind } => (*begin, *kind),
            _ => unreachable!(),
        };
        let perm = (kind / 7) as usize;
        let order = [
            perm % 3,
            (perm + 1 + perm / 3) % 3,
            (perm + 2 - perm / 3) % 3,
        ];
        let ty = kind % 7;
        for i in 0..planes[begin].samples.len() {
            let first = planes[begin].samples[i];
            let mut second = planes[begin + 1].samples[i];
            let mut third = planes[begin + 2].samples[i];
            let values = if ty == 6 {
                let tmp = first.wrapping_sub(third >> 1);
                let g = third.wrapping_add(tmp);
                let b = tmp.wrapping_sub(second >> 1);
                [b.wrapping_add(second), g, b]
            } else {
                if ty & 1 != 0 {
                    third = third.wrapping_add(first);
                }
                if ty >> 1 == 1 {
                    second = second.wrapping_add(first);
                }
                if ty >> 1 == 2 {
                    second = second.wrapping_add(first.wrapping_add(third) >> 1);
                }
                [first, second, third]
            };
            for j in 0..3 {
                planes[begin + order[j]].samples[i] = values[j];
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::super::{predictor::Predictor, tree::Node};
    use super::*;
    #[test]
    fn residual_parser_preserves_literals_and_rejects_nonliteral_input() {
        let options = ModularOptions {
            lz77: false,
            ..Default::default()
        };
        let raw = vec![
            Token::literal(0, 0),
            Token::literal(1, u32::MAX),
            Token::literal(0, 42),
        ];
        assert_eq!(
            parse_residual_tokens(&raw, 2, 17, &options).unwrap(),
            (raw.clone(), false)
        );
        assert!(parse_residual_tokens(&raw, 1, 17, &options).is_err());
        assert_eq!(
            parse_residual_tokens(&raw, 2, 0, &options).unwrap(),
            (raw.clone(), false)
        );
        assert!(parse_residual_tokens(&raw, 0, 17, &options).is_err());
        let mut bad = raw;
        bad[0].is_length = true;
        assert!(parse_residual_tokens(&bad, 2, 17, &options).is_err());
    }
    #[test]
    fn static_channel_leaf_keeps_wire_context_and_scaled_residuals() {
        let tree = Tree::new(vec![
            Node::Split {
                property: 0,
                threshold: 0,
                greater: 1,
                less_equal: 2,
            },
            Node::Leaf {
                predictor: Predictor::Zero,
                offset: -9,
                multiplier: 3,
            },
            Node::Leaf {
                predictor: Predictor::Zero,
                offset: 7,
                multiplier: 5,
            },
        ])
        .unwrap();
        let planes: Vec<_> = (0..2)
            .map(|c| Channel {
                width: 3,
                height: 2,
                hshift: 0,
                vshift: 0,
                samples: (0..6)
                    .map(|i| {
                        if c == 0 {
                            7 + 5 * (i - 3)
                        } else {
                            -9 + 3 * (3 - i)
                        }
                    })
                    .collect(),
            })
            .collect();
        let wp = weighted::Header::default();
        let options = ModularOptions {
            lz77: false,
            previous_channels: 1,
            ..Default::default()
        };
        let (tokens, _) = encode::group_stream(&planes, &tree, &options, &wp, 3).unwrap();
        let layout: Vec<_> = planes
            .iter()
            .map(|p| PlaneLayout {
                width: p.width,
                height: p.height,
                hshift: 0,
                vshift: 0,
            })
            .collect();
        let mut next = tokens.iter();
        let decoded = decode_planes(&layout, &tree, &wp, 3, 1, 12, |context| {
            let token = next.next().unwrap();
            assert_eq!(token.context, context);
            Ok(token.value)
        })
        .unwrap();
        assert_eq!(decoded, planes);
        assert!(next.next().is_none());
    }
    #[test]
    fn multileaf_inverse_recomputes_weighted_and_previous_channel_contexts() {
        let tree = Tree::new(vec![
            Node::Split {
                property: 0,
                threshold: 0,
                greater: 1,
                less_equal: 2,
            },
            Node::Split {
                property: 17,
                threshold: 0,
                greater: 3,
                less_equal: 4,
            },
            Node::Split {
                property: 15,
                threshold: 0,
                greater: 5,
                less_equal: 6,
            },
            Node::Split {
                property: 15,
                threshold: 0,
                greater: 7,
                less_equal: 8,
            },
            Node::Split {
                property: 9,
                threshold: 0,
                greater: 9,
                less_equal: 10,
            },
            Node::Leaf {
                predictor: Predictor::Weighted,
                offset: 3,
                multiplier: 1,
            },
            Node::Leaf {
                predictor: Predictor::Average4,
                offset: -7,
                multiplier: 1,
            },
            Node::Leaf {
                predictor: Predictor::Weighted,
                offset: 17,
                multiplier: 1,
            },
            Node::Leaf {
                predictor: Predictor::Gradient,
                offset: 11,
                multiplier: 1,
            },
            Node::Leaf {
                predictor: Predictor::TopLeft,
                offset: -23,
                multiplier: 1,
            },
            Node::Leaf {
                predictor: Predictor::Weighted,
                offset: 5,
                multiplier: 1,
            },
        ])
        .unwrap();
        let mut used = vec![false; tree.leaves()];
        let wp = weighted::Header::preset(4).unwrap();
        for width in [1, 2, 17] {
            let height = 5;
            let planes: Vec<_> = (0..4)
                .map(|c| Channel {
                    width,
                    height,
                    hshift: 0,
                    vshift: 0,
                    samples: (0..width * height)
                        .map(|i| {
                            if c == 3 {
                                return 0;
                            }
                            [
                                i32::MIN,
                                i32::MAX,
                                -123456789,
                                1,
                                17,
                                (i as i32).wrapping_mul(0x5c123fa1),
                            ][(i + c) % 6]
                        })
                        .collect(),
                })
                .collect();
            let options = ModularOptions {
                lz77: false,
                previous_channels: 1,
                ..Default::default()
            };
            let (tokens, lz) = encode::group_stream(&planes, &tree, &options, &wp, 19).unwrap();
            assert!(!lz);
            let layout: Vec<_> = planes
                .iter()
                .map(|p| PlaneLayout {
                    width: p.width,
                    height: p.height,
                    hshift: p.hshift,
                    vshift: p.vshift,
                })
                .collect();
            let mut tokens = tokens.iter();
            let reconstructed =
                decode_planes(&layout, &tree, &wp, 19, 1, width * height * 4, |context| {
                    let token = tokens.next().expect("one token per reconstructed sample");
                    assert_eq!(context, token.context, "causal context at width {width}");
                    used[context] = true;
                    Ok(token.value)
                })
                .unwrap();
            assert!(tokens.next().is_none());
            assert_eq!(reconstructed, planes, "samples at width {width}");
        }
        assert!(used.iter().filter(|&&seen| seen).count() >= 4);
    }

    #[test]
    fn invalid_layouts_fail_before_residual_reads_or_inverse_mutation() {
        let tree = Tree::new(vec![Node::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
        }])
        .unwrap();
        for (width, height, hshift, vshift) in [
            (0, 1, 0, 0),
            (1, 0, 0, 0),
            (i32::MAX as usize + 1, 1, 0, 0),
            (1, i32::MAX as usize + 1, 0, 0),
            (1, 1, -1, 0),
            (1, 1, 0, -1),
            (1, 1, 32, 0),
            (1, 1, 0, 32),
        ] {
            let layout = [PlaneLayout {
                width,
                height,
                hshift,
                vshift,
            }];
            assert!(
                decode_planes(
                    &layout,
                    &tree,
                    &weighted::Header::default(),
                    0,
                    0,
                    usize::MAX,
                    |_| panic!("invalid geometry must fail before reading")
                )
                .is_err()
            );
        }
        let mut planes = vec![
            Channel {
                width: 2,
                height: 2,
                hshift: 0,
                vshift: 0,
                samples: vec![1],
            };
            3
        ];
        let original = planes.clone();
        assert!(inverse_transforms(&mut planes, &[Transform::Rct(6)]).is_err());
        assert_eq!(planes, original);
        assert!(inverse_transforms(&mut planes, &[]).is_err());
    }

    #[test]
    fn causal_inverse_matches_actual_tokens_wide_hidden_and_weighted() {
        for predictor in [Predictor::Zero, Predictor::Gradient, Predictor::Weighted] {
            let planes: Vec<_> = (0..4)
                .map(|c| Channel {
                    width: 4,
                    height: 3,
                    hshift: 0,
                    vshift: 0,
                    samples: (0..12)
                        .map(|i| {
                            if c == 3 {
                                0
                            } else {
                                [i32::MIN, i32::MAX, 0, -123456789, 1, 17][(i + c) % 6]
                            }
                        })
                        .collect(),
                })
                .collect();
            let tree = Tree::new(vec![Node::Leaf {
                predictor,
                offset: 3,
                multiplier: 1,
            }])
            .unwrap();
            let wp = weighted::Header::preset(0).unwrap();
            let options = ModularOptions {
                lz77: false,
                previous_channels: 1,
                ..ModularOptions::default()
            };
            let (tokens, lz) = encode::group_stream(&planes, &tree, &options, &wp, 0).unwrap();
            assert!(!lz);
            let layout: Vec<_> = planes
                .iter()
                .map(|p| PlaneLayout {
                    width: p.width,
                    height: p.height,
                    hshift: p.hshift,
                    vshift: p.vshift,
                })
                .collect();
            let mut tokens = tokens.iter();
            let decoded = decode_planes(&layout, &tree, &wp, 0, 1, 48, |context| {
                let t = tokens.next().unwrap();
                assert_eq!(t.context, context);
                Ok(t.value)
            })
            .unwrap();
            assert_eq!(decoded, planes);
            assert!(tokens.next().is_none());
            assert!(
                decode_planes(&layout, &tree, &wp, 0, 1, 47, |_| panic!(
                    "allocation cap before callback"
                ))
                .is_err()
            );
        }
    }
    #[test]
    fn rct_inverse_retains_alpha_and_wrapping_words() {
        let original: Vec<_> = (0..4)
            .map(|c| Channel {
                width: 3,
                height: 1,
                hshift: 0,
                vshift: 0,
                samples: vec![i32::MIN + c, i32::MAX - c, c],
            })
            .collect();
        for kind in 0..42 {
            let mut p = original.clone();
            let t = super::super::group::apply_rct(&mut p, 0, 0, kind).unwrap();
            if let Some(t) = t {
                inverse_transforms(&mut p, &[t]).unwrap();
            }
            assert_eq!(p, original, "RCT {kind}");
        }
    }
}
