use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::metadata::Animation;
use rs_jxl::multiframe::{AutoOptions, Reference, Strategy, encode_auto};
use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout};

#[test]
fn crop_preserves_the_reference_for_a_later_signed_update() {
    let mut samples: Vec<u8> = (0..64 * 64)
        .map(|i| ((i * 183 + i / 7) % 251) as u8)
        .collect();
    let mut images = Vec::new();
    for step in 0..4 {
        match step {
            1 | 2 => samples[step * 67] ^= 63,
            3 => samples.iter_mut().for_each(|v| *v = v.wrapping_add(1)),
            _ => {}
        }
        let image = Image::new(64, 64, PixelLayout::Gray, PixelData::U8(samples.clone())).unwrap();
        let mut metadata = image.metadata().clone();
        metadata.animation = Some(Animation::default());
        images.push(image.with_metadata(metadata, vec![]).unwrap());
    }
    let auto = AutoOptions {
        reference: Reference::Previous,
        allow_mask: false,
        ..Default::default()
    };
    let durations = [5, 6, 7, 8];
    let result = encode_auto(&images, &durations, &EncodeOptions::default(), &auto).unwrap();
    let winners: Vec<_> = result
        .frames
        .iter()
        .map(|r| r.candidates[r.winner_index].strategy)
        .collect();
    assert_eq!(
        winners,
        [
            Strategy::Full,
            Strategy::CropPrevious,
            Strategy::CropPrevious,
            Strategy::AddDelta
        ]
    );
    let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
    for ((frame, original), ticks) in decoded.frames.iter().zip(&images).zip(durations) {
        assert_eq!(&frame.pixels, original.pixels());
        assert_eq!(frame.duration_ticks, ticks);
    }
}

#[test]
fn signed_add_chain_refreshes_its_reference_and_preserves_every_frame() {
    let images: Vec<_> = (0..36)
        .map(|frame| {
            let data = (0..32 * 24)
                .map(|i| {
                    let base = ((i * 183 + i / 5) % 256) as u8;
                    if frame % 2 == 0 {
                        base
                    } else if base == 0 || base == 255 {
                        255 - base
                    } else {
                        base ^ 1
                    }
                })
                .collect();
            let image = Image::new(32, 24, PixelLayout::Gray, PixelData::U8(data)).unwrap();
            let mut meta = image.metadata().clone();
            meta.animation = Some(Animation::default());
            image.with_metadata(meta, vec![]).unwrap()
        })
        .collect();
    let policy = AutoOptions {
        reference: Reference::Previous,
        allow_mask: false,
        allow_multi_rect: false,
        ..Default::default()
    };
    let durations: Vec<u32> = (1..=36).collect();
    let result = encode_auto(&images, &durations, &EncodeOptions::default(), &policy).unwrap();
    for (index, report) in result.frames.iter().enumerate() {
        let strategy = report.candidates[report.winner_index].strategy;
        if index == 0 || index == 33 {
            assert_eq!(strategy, Strategy::Full);
        } else {
            assert_eq!(strategy, Strategy::AddDelta);
        }
    }
    let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded.frames.len(), images.len());
    for ((frame, image), ticks) in decoded.frames.iter().zip(&images).zip(durations) {
        assert_eq!(&frame.pixels, image.pixels());
        assert_eq!(frame.duration_ticks, ticks);
    }
}

#[test]
fn terminal_rgba_delta_preserves_alpha_and_hidden_color_from_an_undisplayed_reference() {
    let images: Vec<_> = (0..3)
        .map(|frame| {
            let mut samples = Vec::new();
            for i in 0..32 * 24 {
                let color = ((i * 183 + i / 5) % 256) as u8;
                let delta = u8::from(frame == 2);
                let alpha = [0, 1, 127, 255][i % 4];
                let pixel = match frame {
                    1 => [0, 0, 0, 0],
                    _ => [
                        color ^ delta,
                        !color,
                        color,
                        if i % 5 == 0 { alpha ^ delta } else { alpha },
                    ],
                };
                samples.extend_from_slice(&pixel);
            }
            let image = Image::new(32, 24, PixelLayout::Rgba, PixelData::U8(samples)).unwrap();
            let mut meta = image.metadata().clone();
            meta.animation = Some(Animation::default());
            image.with_metadata(meta, vec![]).unwrap()
        })
        .collect();
    let options = EncodeOptions::default();
    let auto = AutoOptions {
        reference: Reference::First,
        allow_mask: false,
        allow_multi_rect: false,
        prune: false,
        ..Default::default()
    };
    let durations = [0, 5, 7];
    let result = encode_auto(&images, &durations, &options, &auto).unwrap();
    let last = &result.frames[2];
    let winner = &last.candidates[last.winner_index];
    assert_eq!(winner.strategy, Strategy::AddDelta);
    assert_eq!(winner.reference_index, Some(0));
    assert_eq!(winner.trial_pixels, 32 * 24);
    assert_eq!(winner.encoded_frames, 1);
    assert!(
        result.frames[1]
            .candidates
            .iter()
            .all(|c| c.strategy != Strategy::AddDelta)
    );
    let disabled = encode_auto(
        &images,
        &durations,
        &options,
        &AutoOptions {
            allow_add_delta: false,
            ..auto
        },
    )
    .unwrap();
    assert!(result.bytes.len() < disabled.bytes.len());
    assert!(disabled.frames.iter().all(|f| {
        f.candidates
            .iter()
            .all(|c| c.strategy != Strategy::AddDelta)
    }));
    for input in [&result.bytes, &disabled.bytes] {
        let decoded = decode(input, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded.frames.len(), 2);
        for (frame, (image, ticks)) in decoded
            .frames
            .iter()
            .zip(images[1..].iter().zip(&durations[1..]))
        {
            assert_eq!(&frame.pixels, image.pixels());
            assert_eq!(frame.duration_ticks, *ticks);
        }
    }
}
