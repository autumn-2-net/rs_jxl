//! Small parent-run regressions, no external files or decoder process.
use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::modular::{
    encode::FixedTree,
    palette::{
        PaletteMode, PaletteOptions, PaletteStage,
        admission::{Experiment as Admission, Mode},
        counterfactual::{Choice, Experiment, Target},
    },
    predictor::Predictor,
};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode};
use std::sync::Arc;

fn target() -> Target {
    Target {
        stage: PaletteStage::PreRct,
        stream_id: 0,
        ordinal: 0,
    }
}

fn options() -> EncodeOptions {
    EncodeOptions {
        threads: 1,
        group_size_shift: 0,
        modular: ModularOptions {
            rct: None,
            fixed_tree: Some(FixedTree::GradientLeaf),
            lz77: false,
            palette_policy: Some(PaletteOptions {
                palette_colors: 16,
                channel_colors_pre_transform_percent: 0.0,
                channel_colors_percent: 0.0,
                ..PaletteOptions::default()
            }),
            ..ModularOptions::default()
        },
        ..EncodeOptions::default()
    }
}

fn image(layout: PixelLayout, wide: bool) -> Image {
    let colors = [
        [11, 31, 71, 0],
        [19, 43, 89, 0],
        [47, 67, 101, 255],
        [79, 97, 131, 0],
    ];
    let channels = if layout == PixelLayout::Rgb { 3 } else { 4 };
    let samples: Vec<u8> = (0..512)
        .flat_map(|i| colors[(i / 4) % 4][..channels].to_vec())
        .collect();
    let pixels = if wide {
        PixelData::U16(samples.iter().map(|&v| u16::from(v) * 257).collect())
    } else {
        PixelData::U8(samples)
    };
    Image::new(32, 16, layout, pixels).unwrap()
}

fn assert_pixels(bytes: &[u8], image: &Image) {
    let decoded = decode(bytes, &DecodeOptions::default()).unwrap();
    assert!(decoded.complete);
    assert_eq!(decoded.frames.len(), 1);
    assert_eq!(decoded.frames[0].layout, image.layout());
    assert_eq!(&decoded.frames[0].pixels, image.pixels());
}

#[test]
fn shadow_is_byte_neutral_and_veto_is_pixel_exact_for_rgb_rgba_8_16() {
    for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
        for wide in [false, true] {
            let image = image(layout, wide);
            for palette_mode in [
                PaletteMode::Absolute,
                PaletteMode::ExactDelta(Predictor::Left),
                PaletteMode::ExactZero(Predictor::Gradient),
            ] {
                let mut base = options();
                base.modular.palette_policy.as_mut().unwrap().mode = palette_mode;
                let ordinary = encode(&image, &base).unwrap();
                for mode in [Mode::Shadow, Mode::Veto] {
                    let receipt = Arc::new(Admission::new(mode, Some(target())));
                    let mut configured = base.clone();
                    configured
                        .modular
                        .palette_policy
                        .as_mut()
                        .unwrap()
                        .admission = Some(Arc::clone(&receipt));
                    let bytes = encode(&image, &configured).unwrap();
                    assert_pixels(&bytes, &image);
                    if mode == Mode::Shadow {
                        assert_eq!(bytes, ordinary);
                        assert!(receipt.events().iter().all(|event| !event.vetoed));
                    }
                    assert_eq!(receipt.events().len(), 1);
                    let event = receipt.events().remove(0);
                    assert_eq!(event.target, target());
                    if let Some(score) = event.candidate {
                        assert_eq!(score.planes.len(), 2);
                        assert_eq!(score.planes[0].sampled, score.planes[0].population);
                    }
                }
            }
        }
    }
}

#[test]
fn scoring_replays_original_snapshot_without_clobbering_forced_choices() {
    let image = image(PixelLayout::Rgba, true);
    let mut base = options();
    let capture = Arc::new(Experiment::capture(target()));
    base.modular.palette_policy.as_mut().unwrap().counterfactual = Some(Arc::clone(&capture));
    encode(&image, &base).unwrap();
    capture.validate_complete().unwrap();
    let snapshot = capture.snapshot().unwrap();
    for choice in [
        Choice::Skip,
        Choice::Palette(PaletteMode::Absolute),
        Choice::Palette(PaletteMode::ExactDelta(Predictor::Left)),
    ] {
        let mut expected = None;
        for threads in [1, 3] {
            let replay = Arc::new(Experiment::replay(Arc::clone(&snapshot), choice));
            let receipt = Arc::new(Admission::new(Mode::Veto, Some(target())));
            let mut options = base.clone();
            options.threads = threads;
            let policy = options.modular.palette_policy.as_mut().unwrap();
            policy.counterfactual = Some(Arc::clone(&replay));
            policy.admission = Some(Arc::clone(&receipt));
            let bytes = encode(&image, &options).unwrap();
            replay.validate_complete().unwrap();
            assert_pixels(&bytes, &image);
            let record = replay
                .events()
                .into_iter()
                .find(|r| r.target == target())
                .unwrap();
            assert_eq!(record.choice, choice);
            assert!(!record.unavailable);
            let events = receipt.events();
            assert_eq!(events.len(), 1);
            assert!(events[0].forced);
            assert!(!events[0].vetoed);
            if choice == Choice::Skip {
                assert!(record.attempt.is_none());
                assert!(events[0].candidate.is_none());
            } else {
                assert!(events[0].skip.is_some());
                assert!(events[0].candidate.is_some());
            }
            if let Some((previous_bytes, previous_events)) = &expected {
                assert_eq!(&bytes, previous_bytes);
                assert_eq!(&events, previous_events);
            } else {
                expected = Some((bytes, events));
            }
        }
    }
}
