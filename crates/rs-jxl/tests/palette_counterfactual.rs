//! Bounded native encode/decode regressions; no external decoder or fixtures.

use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::palette::{
    PaletteMode, PaletteOptions, PaletteOutcome, PaletteReceipt, PaletteStage,
    counterfactual::{Choice, Experiment, Record, Snapshot, Target},
};
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::{EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout, encode};
use std::sync::Arc;

fn target(stage: PaletteStage, stream_id: usize) -> Target {
    Target {
        stage,
        stream_id,
        ordinal: 0,
    }
}

fn options(limit: i32) -> EncodeOptions {
    EncodeOptions {
        group_size_shift: 0,
        modular: ModularOptions {
            rct: None,
            predictor: Predictor::Gradient,
            fixed_tree: Some(FixedTree::GradientLeaf),
            lz77: false,
            palette_policy: Some(PaletteOptions {
                // One bounded all-channel attempt per stage/stream. No search.
                direct: true,
                palette_colors: limit,
                channel_colors_pre_transform_percent: 0.0,
                channel_colors_percent: 0.0,
                ..PaletteOptions::default()
            }),
            ..ModularOptions::default()
        },
        ..EncodeOptions::default()
    }
}

fn with_experiment(base: &EncodeOptions, experiment: &Arc<Experiment>) -> EncodeOptions {
    let mut options = base.clone();
    options
        .modular
        .palette_policy
        .as_mut()
        .unwrap()
        .counterfactual = Some(Arc::clone(experiment));
    options
}

fn assert_pixels(bytes: &[u8], image: &Image) {
    let decoded = decode(bytes, &DecodeOptions::default()).unwrap();
    assert!(decoded.complete);
    assert_eq!(decoded.frames.len(), 1);
    assert_eq!(
        decoded.info.size,
        (image.width() as usize, image.height() as usize)
    );
    let frame = &decoded.frames[0];
    assert!(frame.complete);
    assert_eq!(frame.layout, image.layout());
    // Native/AsIs output compares every component, including alpha-zero RGB.
    assert_eq!(&frame.pixels, image.pixels());
}

fn roundtrip(image: &Image, options: &EncodeOptions, experiment: &Arc<Experiment>) -> Vec<u8> {
    let bytes = encode(image, &with_experiment(options, experiment)).unwrap();
    experiment.validate_complete().unwrap();
    assert!(!experiment.unavailable());
    assert_pixels(&bytes, image);
    bytes
}

fn selected(experiment: &Experiment, target: Target) -> Record {
    let records: Vec<_> = experiment
        .events()
        .into_iter()
        .filter(|event| event.target == target)
        .collect();
    assert_eq!(records.len(), 1, "selected target must occur exactly once");
    records.into_iter().next().unwrap()
}

fn sorted_events(experiment: &Experiment) -> Vec<Record> {
    let mut events = experiment.events();
    events.sort_by_key(|event| {
        (
            event.target.stage == PaletteStage::PostRct,
            event.target.stream_id,
            event.target.ordinal,
        )
    });
    events
}

fn rgba_or_rgb(layout: PixelLayout) -> Image {
    let colors = [
        [20, 40, 60, 0],
        [30, 60, 90, 0],
        [50, 100, 150, 255],
        [70, 140, 210, 0],
    ];
    let count = if layout == PixelLayout::Rgb { 3 } else { 4 };
    let samples = (0..32)
        .flat_map(|i| colors[i % 4][..count].to_vec())
        .collect();
    Image::new(8, 4, layout, PixelData::U8(samples)).unwrap()
}

fn capture(image: &Image, options: &EncodeOptions, target: Target) -> Arc<Snapshot> {
    let experiment = Arc::new(Experiment::capture(target));
    roundtrip(image, options, &experiment);
    assert_eq!(selected(&experiment, target).choice, Choice::Keep);
    let snapshot = experiment.snapshot().unwrap();
    assert_eq!(snapshot.target(), target);
    snapshot
}

fn assert_choice(record: &Record, choice: Choice) {
    assert_eq!(record.choice, choice);
    assert!(!record.unavailable);
    if choice == Choice::Skip {
        assert!(record.attempt.is_none());
        return;
    }
    let attempt = record.attempt.as_ref().unwrap();
    assert_eq!(attempt.outcome, PaletteOutcome::Accepted);
    if let Choice::Palette(mode) = choice {
        assert_eq!(attempt.mode, mode);
        match mode {
            PaletteMode::ExactDelta(predictor) => {
                assert!(attempt.kernel.exact_delta_attempted);
                assert!(attempt.kernel.exact_delta_produced);
                assert!(!attempt.kernel.absolute_attempted);
                assert!(attempt.deltas > 0);
                assert_eq!(attempt.colors, 0);
                assert_eq!(attempt.predictor, predictor);
            }
            PaletteMode::Absolute => {
                assert!(attempt.kernel.absolute_produced);
                assert!(!attempt.kernel.exact_delta_attempted);
                assert_eq!(attempt.deltas, 0);
            }
            _ => unreachable!("fixture only uses absolute and exact delta"),
        }
    }
}

#[test]
fn global_rgb_and_invisible_rgba_replays_are_pixel_exact() {
    let target = target(PaletteStage::PreRct, 0);
    let options = options(16);
    for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
        let image8 = rgba_or_rgb(layout);
        let PixelData::U8(samples) = image8.pixels() else {
            unreachable!()
        };
        let image16 = Image::new(
            image8.width(),
            image8.height(),
            layout,
            PixelData::U16(
                samples
                    .iter()
                    .map(|&value| u16::from(value) * 257)
                    .collect(),
            ),
        )
        .unwrap();
        for image in [image8, image16] {
            let snapshot = capture(&image, &options, target);
            assert_eq!(
                snapshot.input_channels().len(),
                if layout == PixelLayout::Rgb { 3 } else { 4 }
            );
            assert_eq!(snapshot.details().bits, image.bits());
            assert_eq!(snapshot.base_mode(), PaletteMode::Absolute);
            for choice in [
                Choice::Skip,
                Choice::Palette(PaletteMode::Absolute),
                Choice::Palette(PaletteMode::ExactDelta(Predictor::Left)),
            ] {
                let replay = Arc::new(Experiment::replay(Arc::clone(&snapshot), choice));
                roundtrip(&image, &options, &replay);
                assert_choice(&selected(&replay, target), choice);
            }
        }
    }
}

#[test]
fn local_targets_and_replay_inputs_are_stable_with_one_or_eight_threads() {
    // Six colors globally exceed cap four; each 128x2 AC tile has three colors
    // and at most four Left residuals (initial, +step, -2*step, zero).
    let samples = (0..2)
        .flat_map(|_| {
            (0..256).flat_map(|x| {
                let base = if x < 128 { 20 } else { 70 };
                let value = base + (x % 128 % 3) * 10;
                [value as u8, (value + 20) as u8, (value + 40) as u8]
            })
        })
        .collect();
    let image = Image::new(256, 2, PixelLayout::Rgb, PixelData::U8(samples)).unwrap();
    let serial = options(4);
    let mut parallel = serial.clone();
    parallel.threads = 8;
    let observed_serial = Arc::new(Experiment::observe());
    let observed_parallel = Arc::new(Experiment::observe());
    roundtrip(&image, &serial, &observed_serial);
    roundtrip(&image, &parallel, &observed_parallel);
    let baseline = sorted_events(&observed_serial);
    assert_eq!(baseline, sorted_events(&observed_parallel));
    // One DC group, 17 quant tables: AC stream IDs start at 1 + 3 + 17.
    let targets = [
        target(PaletteStage::PreRct, 0),
        target(PaletteStage::PostRct, 21),
        target(PaletteStage::PostRct, 22),
    ];
    assert_eq!(
        baseline
            .iter()
            .map(|record| record.target)
            .collect::<Vec<_>>(),
        targets
    );
    assert_eq!(
        baseline[0].attempt.as_ref().unwrap().outcome,
        PaletteOutcome::KernelDeclined
    );
    let target = targets[1];
    let snapshot = capture(&image, &serial, target);
    assert_eq!(snapshot.input_channels().len(), 3);
    assert_eq!(
        (
            snapshot.input_channels()[0].width,
            snapshot.input_channels()[0].height
        ),
        (128, 2)
    );
    for choice in [
        Choice::Skip,
        Choice::Palette(PaletteMode::Absolute),
        Choice::Palette(PaletteMode::ExactDelta(Predictor::Left)),
    ] {
        let replay = Arc::new(Experiment::replay(Arc::clone(&snapshot), choice));
        roundtrip(&image, &parallel, &replay);
        assert_choice(&selected(&replay, target), choice);
        assert_eq!(
            sorted_events(&replay)
                .into_iter()
                .filter(|e| e.target != target)
                .collect::<Vec<_>>(),
            baseline
                .iter()
                .filter(|e| e.target != target)
                .cloned()
                .collect::<Vec<_>>(),
            "overriding one stream must not change another stream's attempts"
        );
    }
}

#[test]
fn strict_delta_decline_is_unavailable_not_an_accepted_absolute_fallback() {
    // Existing delta-palette regression: three colors, more than three residuals.
    let colors = [[10, 20, 30], [50, 60, 70], [110, 140, 170]];
    let ids = [0, 0, 1, 0, 2, 1, 2, 2];
    let samples = ids.into_iter().flat_map(|i| colors[i]).collect();
    let image = Image::new(8, 1, PixelLayout::Rgb, PixelData::U8(samples)).unwrap();
    let mut options = options(3);
    let mode = PaletteMode::ExactDelta(Predictor::Left);
    options.modular.palette_policy.as_mut().unwrap().mode = mode;
    let target = target(PaletteStage::PreRct, 0);
    let capture = Arc::new(Experiment::capture(target));
    roundtrip(&image, &options, &capture);
    let ordinary = selected(&capture, target).attempt.unwrap();
    assert!(ordinary.kernel.exact_delta_attempted);
    assert!(!ordinary.kernel.exact_delta_produced);
    assert!(ordinary.kernel.absolute_produced);
    assert_eq!(ordinary.outcome, PaletteOutcome::Accepted);

    let snapshot = capture.snapshot().unwrap();
    let choice = Choice::Palette(mode);
    let replay = Arc::new(Experiment::replay(Arc::clone(&snapshot), choice));
    let error = encode(&image, &with_experiment(&options, &replay)).unwrap_err();
    assert!(matches!(error, Error::Encoding(ref message)
        if message == "counterfactual palette kernel unavailable at fixed input"));
    assert!(replay.unavailable());
    // Input validation succeeded; availability is a separate result of encoding.
    replay.validate_complete().unwrap();
    let record = selected(&replay, target);
    assert_eq!(record.choice, choice);
    assert!(record.unavailable);
    let attempt = record.attempt.unwrap();
    assert!(!attempt.kernel.exact_delta_produced);
    assert!(
        attempt.kernel.absolute_produced,
        "ordinary fallback was possible but not admitted"
    );
    assert_eq!(attempt.outcome, PaletteOutcome::KernelDeclined);

    let replay = Arc::new(Experiment::replay(
        snapshot,
        Choice::Palette(PaletteMode::Absolute),
    ));
    roundtrip(&image, &options, &replay);
    assert_choice(
        &selected(&replay, target),
        Choice::Palette(PaletteMode::Absolute),
    );
}

#[test]
fn default_observation_does_not_change_encoding_or_capture_samples() {
    let image = rgba_or_rgb(PixelLayout::Rgba);
    let mut options = options(16);
    let policy = PaletteOptions::default();
    assert!(policy.counterfactual.is_none());
    options.modular.palette_policy = Some(policy);
    let baseline = encode(&image, &options).unwrap();
    assert_pixels(&baseline, &image);
    let observation = Arc::new(Experiment::observe());
    let receipt = Arc::new(PaletteReceipt::default());
    options.modular.palette_policy.as_mut().unwrap().receipt = Some(Arc::clone(&receipt));
    let observed = roundtrip(&image, &options, &observation);
    assert_eq!(observed, baseline, "observation must be behavior-neutral");
    assert!(observation.snapshot().is_none());
    let events = observation.events();
    assert!(!events.is_empty());
    assert!(
        events
            .iter()
            .all(|e| e.choice == Choice::Keep && !e.unavailable)
    );
    assert_eq!(
        events
            .into_iter()
            .map(|e| e.attempt.unwrap())
            .collect::<Vec<_>>(),
        receipt.snapshot()
    );
}

#[test]
fn wrong_hidden_input_and_missing_targets_are_not_valid_replays() {
    let image = rgba_or_rgb(PixelLayout::Rgba);
    let options = options(16);
    let target = target(PaletteStage::PreRct, 0);
    let snapshot = capture(&image, &options, target);
    let PixelData::U8(mut changed) = image.pixels().clone() else {
        unreachable!()
    };
    assert_eq!(changed[3], 0);
    changed[0] += 1; // Only invisible red changes; rendered equivalence is insufficient.
    let changed = Image::new(8, 4, PixelLayout::Rgba, PixelData::U8(changed)).unwrap();
    let replay = Arc::new(Experiment::replay(Arc::clone(&snapshot), Choice::Skip));
    let error = encode(&changed, &with_experiment(&options, &replay)).unwrap_err();
    assert!(matches!(error, Error::InvalidInput(ref message)
        if message.contains("input channels or samples differ from snapshot")));
    assert!(replay.validate_complete().is_err());
    assert!(!replay.unavailable());
    assert!(
        replay.events().is_empty(),
        "mismatch must precede choice/recording"
    );

    let missing = Arc::new(Experiment::capture(Target {
        stream_id: usize::MAX,
        ..target
    }));
    let bytes = encode(&image, &with_experiment(&options, &missing)).unwrap();
    assert_pixels(&bytes, &image);
    assert!(
        matches!(missing.validate_complete(), Err(Error::InvalidInput(message))
        if message.contains("selected target was not visited"))
    );
    assert!(missing.snapshot().is_none());

    // Encoding does not implicitly validate experiments: disabling all attempts
    // succeeds, but explicit completion validation rejects an unvisited replay.
    let replay = Arc::new(Experiment::replay(snapshot, Choice::Skip));
    let mut disabled = options.clone();
    disabled
        .modular
        .palette_policy
        .as_mut()
        .unwrap()
        .palette_colors = 0;
    let bytes = encode(&image, &with_experiment(&disabled, &replay)).unwrap();
    assert_pixels(&bytes, &image);
    assert!(replay.events().is_empty());
    assert!(
        matches!(replay.validate_complete(), Err(Error::InvalidInput(message))
        if message.contains("selected target was not visited"))
    );
    assert!(!replay.unavailable());
}
