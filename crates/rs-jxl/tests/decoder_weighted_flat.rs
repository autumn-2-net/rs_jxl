//! Flat weighted prediction must still update history when a run breaks.
//! Original integers, including invisible RGB, are the roundtrip oracle.

use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::metadata::BitDepth;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::modular::tree::{Node, Tree};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode};

fn leaf(predictor: Predictor) -> Node {
    Node::Leaf {
        predictor,
        offset: 0,
        multiplier: 1,
    }
}

fn tree(route: usize) -> Tree {
    Tree::new(match route {
        0 => vec![leaf(Predictor::Weighted)],
        1 => vec![
            Node::Split {
                property: 15,
                threshold: 0,
                greater: 1,
                less_equal: 2,
            },
            leaf(Predictor::Weighted),
            leaf(Predictor::Weighted),
        ],
        _ => vec![
            Node::Split {
                property: 9,
                threshold: 127,
                greater: 1,
                less_equal: 2,
            },
            leaf(Predictor::Weighted),
            leaf(Predictor::Left),
        ],
    })
    .unwrap()
}

#[test]
fn flat_runs_and_breaks_preserve_samples_across_weighted_decoder_routes() {
    for bits in [8u8, 16, 31] {
        let max = (1u32 << bits) - 1;
        for width in [1u32, 17] {
            let values: Vec<u32> = (0..69)
                .flat_map(|y| {
                    (0..width).flat_map(move |x| {
                        let base = match y {
                            0..=17 => 0,
                            18..=34 => max,
                            35..=51 => max / 2,
                            _ => (x * 197 + y * 40503) & max,
                        };
                        // Abrupt run breaks and zero alpha with nonzero RGB.
                        let r = if x == width / 2 && y % 17 == 0 {
                            !base & max
                        } else {
                            base
                        };
                        [r, max - base, base, if y < 52 { 0 } else { max }]
                    })
                })
                .collect();
            let pixels = match bits {
                8 => PixelData::U8(values.iter().map(|&v| v as u8).collect()),
                16 => PixelData::U16(values.iter().map(|&v| v as u16).collect()),
                _ => PixelData::U32(values),
            };
            let source = Image::new(width, 69, PixelLayout::Rgba, pixels).unwrap();
            let mut metadata = source.metadata().clone();
            metadata.bit_depth = BitDepth::new(bits).unwrap();
            for extra in &mut metadata.extra_channels {
                extra.bit_depth = metadata.bit_depth;
            }
            let source = source.with_metadata(metadata, vec![]).unwrap();
            for preset in 0..5 {
                for route in 0..3 {
                    for coding in [Coding::Ans, Coding::Prefix] {
                        let options = EncodeOptions {
                            modular: ModularOptions {
                                custom_tree: Some(tree(route)),
                                weighted_preset: preset,
                                rct: None,
                                lz77: false,
                                entropy: coding,
                                ..Default::default()
                            },
                            ..Default::default()
                        };
                        let encoded = encode(&source, &options).unwrap();
                        let decoded = decode(&encoded, &DecodeOptions::default()).unwrap();
                        assert!(decoded.complete);
                        assert_eq!(
                            decoded.frames[0].pixels,
                            *source.pixels(),
                            "bits={bits}, width={width}, preset={preset}, route={route}, coding={coding:?}"
                        );
                    }
                }
            }
        }
    }
}
