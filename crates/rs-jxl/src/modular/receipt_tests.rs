use super::*;
use crate::modular::encode::FixedTree;
use crate::modular::encode::receipt::{GlobalTuplePalette, ReceiptCollector, ReceiptRoute};
use crate::modular::predictor::Predictor;
use crate::{PixelData, PixelLayout};
use std::sync::Arc;

fn options() -> ModularOptions {
    ModularOptions {
        rct: None,
        predictor: Predictor::Zero,
        fixed_tree: Some(FixedTree::Zero),
        lz77: false,
        palette_policy: Some(PaletteOptions {
            palette_colors: 8,
            force_palette: true,
            channel_colors_pre_transform_percent: 0.0,
            channel_colors_percent: 0.0,
            ..PaletteOptions::default()
        }),
        ..ModularOptions::default()
    }
}

fn rgba(width: u32, height: u32, varying_alpha: bool) -> Image {
    let pixels = (0..width as usize * height as usize)
        .flat_map(|i| {
            let c = (i % 2) as u8;
            [
                7 + c * 13,
                19 + c * 17,
                31 + c * 23,
                if varying_alpha { i as u8 } else { 255 },
            ]
        })
        .collect();
    Image::new(width, height, PixelLayout::Rgba, PixelData::U8(pixels)).unwrap()
}

fn attach(options: &mut ModularOptions) -> Arc<ReceiptCollector> {
    let collector = Arc::new(ReceiptCollector::default());
    options.representation_receipt = Some(Arc::clone(&collector));
    collector
}

fn prepared(image: &Image, options: &ModularOptions) -> PreparedLayout {
    prepare_layout(image, options, 0, &Passes::default(), false).unwrap()
}

fn encoded(image: &Image, options: &ModularOptions) -> Vec<Vec<u8>> {
    encode_layout(
        image,
        options,
        0,
        &Passes::default(),
        false,
        true,
        1 << 20,
        None,
    )
    .unwrap()
}

#[test]
fn receipt_global_tuple_uses_actual_input_dimensions_and_does_not_change_output() {
    assert!(ModularOptions::default().representation_receipt.is_none());
    for (width, height) in [(33, 7), (33, 2)] {
        // The second Image stands in for a crop. Its dimensions, not those of
        // an unavailable original Image, must appear in its own receipt.
        let image = rgba(width, height, false);
        let mut options = options();
        let expected = encoded(&image, &options);
        let collector = attach(&mut options);
        let actual = encoded(&image, &options);
        assert_eq!(actual, expected);
        let receipt = collector.get().unwrap();
        assert_eq!(receipt.route, ReceiptRoute::FullImage);
        assert_eq!((receipt.input_width, receipt.input_height), (width, height));
        assert_eq!(receipt.input_channels, 4);
        assert_eq!(receipt.spatial_channels, 1);
        assert_eq!(receipt.meta_channels, 1);
        assert_eq!(
            receipt.global_tuple_palettes,
            [GlobalTuplePalette {
                input_channels: 0..4,
                colors: 2,
                deltas: 0,
                predictor: Predictor::Zero,
            }]
        );
    }
}

#[test]
fn receipt_all_minus_one_covers_rgb_but_not_alpha() {
    let image = rgba(33, 7, true);
    let mut options = options();
    let collector = attach(&mut options);
    let prepared = prepared(&image, &options);
    assert!(matches!(
        prepared.global.transforms[0],
        Transform::Palette { channels: 3, .. }
    ));
    let receipt = collector.get().unwrap();
    assert_eq!(receipt.input_channels, 4);
    assert_eq!(receipt.spatial_channels, 2);
    assert_eq!(receipt.meta_channels, 1);
    assert_eq!(receipt.global_tuple_palettes.len(), 1);
    assert_eq!(receipt.global_tuple_palettes[0].input_channels, 0..3);
}

#[test]
fn receipt_actual_delta_palette_keeps_zero_colors_and_nonzero_deltas() {
    let image = rgba(33, 7, false);
    let mut options = options();
    options.palette_policy.as_mut().unwrap().mode = PaletteMode::ExactDelta(Predictor::Left);
    let collector = attach(&mut options);
    let p = prepared(&image, &options);
    let Transform::Palette {
        colors,
        deltas,
        predictor,
        ..
    } = &p.global.transforms[0]
    else {
        panic!("expected accepted global delta palette");
    };
    assert_eq!(*colors, 0);
    assert!(*deltas > 0);
    assert_eq!(*predictor, Predictor::Left);
    let receipt = collector.get().unwrap();
    assert_eq!(receipt.spatial_channels, 1);
    assert_eq!(
        receipt.global_tuple_palettes,
        [GlobalTuplePalette {
            input_channels: 0..4,
            colors: *colors,
            deltas: *deltas,
            predictor: *predictor,
        }]
    );
}

#[test]
fn receipt_declined_and_single_channel_palettes_are_not_tuples() {
    let image = rgba(33, 7, true);
    let mut direct = options();
    direct.palette_policy.as_mut().unwrap().direct = true;
    let collector = attach(&mut direct);
    let p = prepared(&image, &direct);
    assert!(p.global.transforms.is_empty());
    let receipt = collector.get().unwrap();
    assert!(receipt.global_tuple_palettes.is_empty());
    assert_eq!((receipt.spatial_channels, receipt.meta_channels), (4, 0));

    let image = rgba(33, 7, false);
    let mut singles = options();
    let policy = singles.palette_policy.as_mut().unwrap();
    policy.force_palette = false;
    policy.palette_colors = 0;
    policy.speed_tier = 4;
    policy.channel_colors_pre_transform_percent = 100.0;
    let collector = attach(&mut singles);
    let p = prepared(&image, &singles);
    assert!(!p.global.transforms.is_empty());
    assert!(
        p.global
            .transforms
            .iter()
            .all(|t| matches!(t, Transform::Palette { channels: 1, .. }))
    );
    let receipt = collector.get().unwrap();
    assert!(receipt.global_tuple_palettes.is_empty());
    assert_eq!(receipt.spatial_channels, 4);
    assert_eq!(receipt.meta_channels, p.global.transforms.len());

    let gray = Image::new(
        33,
        7,
        PixelLayout::Gray,
        PixelData::U8((0..231).map(|i| if i % 2 == 0 { 7 } else { 91 }).collect()),
    )
    .unwrap();
    direct.palette_policy.as_mut().unwrap().palette_colors = 8;
    let collector = attach(&mut direct);
    prepared(&gray, &direct);
    let receipt = collector.get().unwrap();
    assert!(receipt.global_tuple_palettes.is_empty());
    assert_eq!((receipt.spatial_channels, receipt.meta_channels), (1, 1));
}

#[test]
fn receipt_groupwise_excludes_local_tuples_and_fallback_reports_actual_full_route() {
    let image = rgba(1025, 1, false);
    let mut options = options();
    options.palette_policy.as_mut().unwrap().groupwise = true;
    assert_eq!(
        groupwise_decision(&image, &options, 0, &Passes::default(), false, false).unwrap(),
        GroupwiseDecision::FrameAreas
    );
    let collector = attach(&mut options);
    let area = layout::frame_areas(1025, 1, 0).unwrap()[0];
    let p = prepare_frame_area(&image, &options, 0, area).unwrap();
    assert!(p.global.transforms.is_empty());
    assert!(p.streams.iter().any(|g| {
        g.transforms
            .iter()
            .any(|t| matches!(t, Transform::Palette { channels: 4, .. }))
    }));
    // Area preparation alone must never publish a whole-Image receipt.
    assert!(collector.get().is_none());
    encoded(&image, &options);
    let receipt = collector.get().unwrap();
    assert_eq!(receipt.route, ReceiptRoute::Groupwise);
    assert_eq!((receipt.input_width, receipt.input_height), (1025, 1));
    assert_eq!((receipt.spatial_channels, receipt.meta_channels), (4, 0));
    assert!(receipt.global_tuple_palettes.is_empty());

    let small = rgba(17, 3, false);
    let collector = attach(&mut options);
    encoded(&small, &options);
    let receipt = collector.get().unwrap();
    assert_eq!(receipt.route, ReceiptRoute::FullImage);
    assert_eq!(receipt.spatial_channels, 1);
    assert_eq!(receipt.global_tuple_palettes[0].input_channels, 0..4);
}

#[test]
fn receipt_spatial_count_is_before_partition_and_local_squeeze() {
    let image = rgba(257, 3, false);
    let mut options = options();
    options.squeeze.push(true);
    let collector = attach(&mut options);
    let p = prepared(&image, &options);
    assert!(p.streams.iter().any(|g| g.planes.len() > 1));
    let receipt = collector.get().unwrap();
    assert_eq!((receipt.spatial_channels, receipt.meta_channels), (1, 1));
    assert_eq!(receipt.global_tuple_palettes[0].input_channels, 0..4);
}

#[test]
fn receipt_channel_coverage_tracks_metadata_insertions_and_keeps_delta_fields() {
    let palette = |begin, channels, colors, deltas, predictor| Transform::Palette {
        begin,
        channels,
        colors,
        deltas,
        predictor,
    };
    let script = [
        palette(0, 1, 2, 0, Predictor::Zero),
        palette(2, 3, 0, 5, Predictor::Left),
        palette(4, 1, 3, 0, Predictor::Zero),
    ];
    let receipt = RepresentationReceipt::full_image(9, 7, 5, &script, 3, 3).unwrap();
    assert_eq!(
        receipt.global_tuple_palettes,
        [GlobalTuplePalette {
            input_channels: 1..4,
            colors: 0,
            deltas: 5,
            predictor: Predictor::Left,
        }]
    );
    assert_eq!((receipt.spatial_channels, receipt.meta_channels), (3, 3));
}

#[test]
fn receipt_collectors_are_isolated_and_reject_reused_options() {
    let a = Arc::new(ReceiptCollector::default());
    let b = Arc::new(ReceiptCollector::default());
    std::thread::scope(|scope| {
        for (collector, width) in [(&a, 17), (&b, 31)] {
            scope.spawn(move || {
                collector.begin().unwrap();
                collector
                    .publish(RepresentationReceipt::groupwise(width, 3, 4))
                    .unwrap();
            });
        }
    });
    assert_eq!(a.get().unwrap().input_width, 17);
    assert_eq!(b.get().unwrap().input_width, 31);
    assert!(a.begin().is_err());

    let mut options = options();
    let collector = attach(&mut options);
    let cloned = options.clone();
    assert!(Arc::ptr_eq(
        cloned.representation_receipt.as_ref().unwrap(),
        &collector
    ));
    let image = rgba(17, 3, false);
    prepared(&image, &options);
    let snapshot = collector.get().unwrap().clone();
    assert!(prepare_layout(&image, &cloned, 0, &Passes::default(), false).is_err());
    assert_eq!(collector.get(), Some(&snapshot));
}

#[test]
fn receipt_failed_preparation_has_no_snapshot_and_requires_new_collector() {
    let image = rgba(17, 3, false);
    let mut options = options();
    let collector = attach(&mut options);
    assert!(prepare_layout(&image, &options, 4, &Passes::default(), false).is_err());
    assert!(collector.get().is_none());
    assert!(prepare_layout(&image, &options, 0, &Passes::default(), false).is_err());
    let fresh = attach(&mut options);
    prepared(&image, &options);
    assert!(fresh.get().is_some());
    assert!(collector.get().is_none());
}
