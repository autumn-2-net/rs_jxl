use super::*;
use crate::modular::{frame, orchestration, palette::PaletteOptions};
use crate::{PixelData, PixelLayout};
use std::sync::Arc;

fn options() -> ModularOptions {
    ModularOptions {
        rct: None,
        predictor: Predictor::Zero,
        fixed_tree: Some(FixedTree::Zero),
        lz77: false,
        entropy: Coding::Prefix,
        ..ModularOptions::default()
    }
}

fn prepared() -> Prepared {
    Prepared {
        planes: vec![
            Channel {
                width: 3,
                height: 1,
                hshift: -1,
                vshift: -1,
                samples: vec![7, 11, 19],
            },
            Channel {
                width: 32,
                height: 4,
                hshift: 0,
                vshift: 0,
                samples: (0..128).map(|i| i % 3).collect(),
            },
        ],
        transforms: vec![Transform::Palette {
            begin: 0,
            channels: 1,
            colors: 3,
            deltas: 0,
            predictor: Predictor::Zero,
        }],
        wp: weighted::Header::preset(0).unwrap(),
        stream_id: 21,
    }
}

fn measured_headers(g: &Prepared, options: &ModularOptions) -> (usize, usize) {
    let tree = predefined(g.planes.iter(), options, 8).unwrap().unwrap();
    let (tokens, lz77) = group_stream(&g.planes, &tree, options, &g.wp, g.stream_id).unwrap();
    let model = Model::build_with_options(
        &[tokens],
        tree.leaves() + usize::from(lz77),
        lz77,
        options.entropy,
        &options.entropy_options,
    )
    .unwrap();
    let mut writer = BitWriter::new(1 << 20);
    writer.write(3, 5).unwrap();
    write_tree(&tree, &mut writer, options).unwrap();
    let tree_bits = writer.bits_written() - 3;
    let before = writer.bits_written();
    model.write_header(&mut writer).unwrap();
    (tree_bits, writer.bits_written() - before)
}

#[test]
fn cost_receipt_measures_only_emitted_headers_in_both_local_writers() {
    assert!(ModularOptions::default().model_cost_receipt.is_none());
    let g = prepared();
    for coding in [Coding::Prefix, Coding::Ans] {
        for source_tree_entropy in [false, true] {
            let mut options = options();
            options.entropy = coding;
            options.source_tree_entropy = source_tree_entropy;
            let expected =
                frame::encode_prepared_independent(&g, &options, 1 << 20, true, None, 8).unwrap();
            let receipt = Arc::new(receipt_cost::ModelCostReceipt::default());
            options.model_cost_receipt = Some(Arc::clone(&receipt));
            // Model cost searches and direct reference tree writes must not count.
            let (tree, model) = measured_headers(&g, &options);
            assert_eq!(receipt.bits(), 0);
            let actual =
                frame::encode_prepared_independent(&g, &options, 1 << 20, true, None, 8).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(receipt.tree_bits(), tree);
            assert_eq!(receipt.entropy_header_bits(), model);
            assert!(receipt.bits() < actual.len() * 8);

            let mut writer = BitWriter::new(1 << 20);
            writer.write(5, 17).unwrap();
            write_prepared_stream(
                &g.planes,
                &g.transforms,
                &options,
                &mut writer,
                g.stream_id,
                8,
            )
            .unwrap();
            assert_eq!(receipt.bits(), 2 * (tree + model));
            options.model_cost_receipt = None;
            let mut reference = BitWriter::new(1 << 20);
            reference.write(5, 17).unwrap();
            write_prepared_stream(
                &g.planes,
                &g.transforms,
                &options,
                &mut reference,
                g.stream_id,
                8,
            )
            .unwrap();
            assert_eq!(writer.bits_written(), reference.bits_written());
            writer.zero_pad_to_byte().unwrap();
            reference.zero_pad_to_byte().unwrap();
            assert_eq!(
                writer.into_bytes().unwrap(),
                reference.into_bytes().unwrap()
            );
        }
    }
}

#[test]
fn cost_receipt_shared_global_payload_is_not_model_cost() {
    let mut options = options();
    let encode = |options: &ModularOptions| {
        frame::encode_prepared_shared(
            Some(prepared()),
            vec![prepared(), prepared()],
            options,
            1 << 20,
            None,
            8,
        )
        .unwrap()
    };
    let expected = encode(&options);
    let receipt = Arc::new(receipt_cost::ModelCostReceipt::default());
    options.model_cost_receipt = Some(Arc::clone(&receipt));
    let actual = encode(&options);
    assert_eq!(actual, expected);
    let (tree, _) = measured_headers(&prepared(), &options);
    assert_eq!(receipt.tree_bits(), tree);
    assert!(receipt.entropy_header_bits() > 0);
    // DCGlobal contains BOTH palette samples and spatial indices here.
    assert!(receipt.bits() < actual.0.len() * 8);
}

#[test]
fn cost_receipt_accumulates_concurrent_independent_writers() {
    let mut options = options();
    let g = prepared();
    let (tree, model) = measured_headers(&g, &options);
    let receipt = Arc::new(receipt_cost::ModelCostReceipt::default());
    options.model_cost_receipt = Some(Arc::clone(&receipt));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                frame::encode_prepared_independent(&g, &options, 1 << 20, false, None, 8).unwrap();
            });
        }
    });
    assert_eq!(receipt.tree_bits(), 4 * tree);
    assert_eq!(receipt.entropy_header_bits(), 4 * model);
    assert_eq!(receipt.bits(), 4 * (tree + model));
    assert_eq!(receipt_cost::ModelCostReceipt::default().bits(), 0);
}

#[test]
fn cost_receipt_groupwise_counts_each_local_tree_once() {
    let image = Image::new(
        1152,
        1,
        PixelLayout::Gray,
        PixelData::U8((0..1152).map(|i| (i % 251) as u8).collect()),
    )
    .unwrap();
    let mut options = options();
    options.palette_policy = Some(PaletteOptions {
        palette_colors: 0,
        channel_colors_percent: 0.0,
        channel_colors_pre_transform_percent: 0.0,
        groupwise: true,
        ..PaletteOptions::default()
    });
    let encode = |options: &ModularOptions| {
        orchestration::encode_layout_trial(
            &image,
            options,
            0,
            &Default::default(),
            false,
            true,
            1 << 20,
            None,
            None,
        )
        .unwrap()
        .into_complete()
        .unwrap()
    };
    let expected = encode(&options);
    let receipt = Arc::new(receipt_cost::ModelCostReceipt::default());
    options.model_cost_receipt = Some(Arc::clone(&receipt));
    assert_eq!(encode(&options), expected);
    let (tree, _) = measured_headers(&prepared(), &options);
    assert_eq!(receipt.tree_bits(), 9 * tree);
    assert!(receipt.entropy_header_bits() > 0);
    assert!(receipt.bits() < expected.iter().map(Vec::len).sum::<usize>() * 8);
}
