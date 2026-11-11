use super::*;
use crate::decode::{DecodeOptions, decode};
use crate::modular::encode::FixedTree;
use crate::modular::palette_groups::{Grouping, Receipt};
use crate::{EncodeOptions, PixelData, PixelLayout};
use std::sync::Arc;

fn options(grouping: Grouping, cap: i32) -> EncodeOptions {
    let mut options = EncodeOptions::default();
    options.threads = 2;
    options.group_size_shift = 0;
    options.modular.fixed_tree = Some(FixedTree::GradientLeaf);
    options.modular.rct = Some(6);
    options.modular.rct_candidates = Some(19);
    options.modular.palette_policy = Some(PaletteOptions {
        grouping: Some(grouping),
        palette_colors: cap,
        ..Default::default()
    });
    options.modular.palette_group_receipt = Some(Arc::new(Receipt::default()));
    options
}

#[test]
fn grouped_palette_wire_roundtrip_rgb_rgba_depths_and_multiple_groups() {
    for bits in [8, 16] {
        for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
            let max = (1u32 << bits) - 1;
            let words: Vec<_> = (0..129 * 9)
                .flat_map(|i| {
                    let n = i % 32;
                    [
                        n * 3,
                        max - n * 5,
                        n * 7,
                        if n % 3 == 0 { 0 } else { n * 2 },
                    ]
                    .into_iter()
                    .take(layout.channels())
                })
                .collect();
            let pixels = if bits == 8 {
                PixelData::U8(words.iter().map(|&v| v as u8).collect())
            } else {
                PixelData::U16(words.iter().map(|&v| v as u16).collect())
            };
            let image = Image::new(129, 9, layout, pixels).unwrap();
            for grouping in [
                Grouping::Rg,
                Grouping::YCo,
                Grouping::YCg,
                Grouping::CoCg,
                Grouping::YaCoCg,
            ] {
                let options = options(grouping, 64);
                let bytes = crate::encode(&image, &options).unwrap();
                let restored = decode(&bytes, &DecodeOptions::default()).unwrap();
                assert_eq!(
                    restored.frames[0].pixels,
                    *image.pixels(),
                    "{bits} {layout:?} {grouping:?}"
                );
                let receipt = options
                    .modular
                    .palette_group_receipt
                    .as_ref()
                    .unwrap()
                    .snapshot();
                assert_eq!(receipt.attempts, 1);
                let accepted = grouping != Grouping::YaCoCg || layout == PixelLayout::Rgba;
                assert_eq!(receipt.applied, usize::from(accepted));
                assert_eq!(
                    receipt.tables,
                    if !accepted {
                        0
                    } else if grouping == Grouping::YaCoCg {
                        2
                    } else {
                        1
                    }
                );
            }
        }
    }
}

#[test]
fn grouped_palette_decline_keeps_normal_encoding_available() {
    let image = Image::new(
        33,
        7,
        PixelLayout::Rgb,
        PixelData::U8((0..33 * 7 * 3).map(|i| (i * 17) as u8).collect()),
    )
    .unwrap();
    let options = options(Grouping::CoCg, 2);
    let bytes = crate::encode(&image, &options).unwrap();
    assert_eq!(
        decode(&bytes, &DecodeOptions::default()).unwrap().frames[0].pixels,
        *image.pixels()
    );
    assert_eq!(
        options
            .modular
            .palette_group_receipt
            .as_ref()
            .unwrap()
            .snapshot()
            .applied,
        0
    );
}

#[test]
fn grouped_palette_configuration_is_explicit_and_cannot_be_silently_ignored() {
    assert!(PaletteOptions::default().grouping.is_none());
    for field in 0..6 {
        let mut policy = options(Grouping::Rg, 64).modular.palette_policy.unwrap();
        match field {
            0 => policy.groupwise = true,
            1 => policy.responsive = true,
            2 => policy.direct = true,
            3 => policy.palette_colors = 0,
            4 => policy.force_palette = true,
            _ => {
                policy.mode =
                    PaletteMode::ExactDelta(crate::modular::predictor::Predictor::Gradient)
            }
        }
        assert!(policy.validate().is_err());
    }
    let mut options = options(Grouping::Rg, 64);
    options.modular.representation_receipt = Some(Arc::new(Default::default()));
    assert!(options.modular.validate().is_err());
}
