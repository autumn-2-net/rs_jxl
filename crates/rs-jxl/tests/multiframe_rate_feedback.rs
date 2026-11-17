//! Recent complete-Full rate feedback is an additional geometry proposal, not
//! a replacement for the original first-frame estimate or an encoded-size oracle.
use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::frame::{BlendInfo, Frame, FrameOptions};
use rs_jxl::metadata::{Animation, BitDepth, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::multiframe::{
    AnalysisOptions, AutoOptions, Reference, Strategy, analyze, crop, encode_auto,
};
use rs_jxl::{EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout, encode_frames};

const SIDE: usize = 128;
const DURATIONS: [u32; 3] = [2, 3, 5];

fn options() -> EncodeOptions {
    EncodeOptions {
        threads: 1,
        modular: ModularOptions {
            rct: None,
            lz77: false,
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn animated(image: Image, extras: Vec<PixelData>) -> Image {
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation::default());
    image.with_metadata(metadata, extras).unwrap()
}

fn fixture() -> Vec<Image> {
    let mut state = 0x793a_5621u32;
    let random: Vec<u8> = (0..SIDE * SIDE * 3)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect();
    let mut changed = random.clone();
    // 64 occupied tiles exceed the legacy 16-rectangle limit. Tight row
    // partitions are cheap at the recent random frame's measured rate, while
    // the solid first frame makes a large bounding rectangle look inexpensive.
    for y in (8..SIDE).step_by(16) {
        for x in (8..SIDE).step_by(16) {
            changed[(y * SIDE + x) * 3] ^= 0x7f;
        }
    }
    [vec![0; random.len()], random, changed]
        .into_iter()
        .map(|values| {
            animated(
                Image::new(
                    SIDE as u32,
                    SIDE as u32,
                    PixelLayout::Rgb,
                    PixelData::U8(values),
                )
                .unwrap(),
                vec![],
            )
        })
        .collect()
}

fn auto(prune: bool) -> AutoOptions {
    AutoOptions {
        adaptive_splits: false,
        adaptive_rate_feedback: true,
        reference: Reference::Previous,
        allow_mask: false,
        tile_size: 16,
        max_rectangles: 16,
        prune,
        ..Default::default()
    }
}

fn same_words(actual: &PixelData, expected: &PixelData) {
    match (actual, expected) {
        (PixelData::F32(a), PixelData::F32(b)) => assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        ),
        _ => assert_eq!(actual, expected),
    }
}

fn verify(bytes: &[u8], images: &[Image]) {
    let decoded = decode(bytes, &DecodeOptions::default()).unwrap();
    assert!(decoded.complete);
    assert_eq!(decoded.frames.len(), images.len());
    assert_eq!(
        decoded.info.bit_depth.bits_per_sample(),
        u32::from(images[0].bits())
    );
    for ((frame, image), duration) in decoded.frames.iter().zip(images).zip(DURATIONS) {
        assert!(frame.complete);
        assert_eq!(
            (frame.width, frame.height),
            (image.width() as usize, image.height() as usize)
        );
        assert_eq!(frame.layout, image.layout());
        assert_eq!(frame.duration_ticks, duration);
        same_words(&frame.pixels, image.pixels());
        // Alpha is also returned as an authoritative extra plane. Explicit
        // non-alpha extras below have no interleaved duplicate.
        if image.layout().has_alpha() {
            let PixelData::U8(values) = image.pixels() else {
                unreachable!()
            };
            assert_eq!(
                frame.extra_planes,
                [PixelData::U8(
                    values.chunks_exact(4).map(|p| p[3]).collect()
                )]
            );
        } else {
            assert_eq!(frame.extra_planes.len(), image.extra_planes().len());
            for (actual, expected) in frame.extra_planes.iter().zip(image.extra_planes()) {
                same_words(actual, expected);
            }
        }
    }
}

fn static_geometry_baseline(images: &[Image], options: &EncodeOptions, anchor: f64) -> Vec<u8> {
    let geometry = analyze(
        images,
        &AnalysisOptions {
            reference: Reference::Previous,
            allow_mask: false,
            tile_size: 16,
            max_rectangles: 16,
            rectangle_bytes_per_pixel: anchor,
            ..Default::default()
        },
    )
    .unwrap();
    // Encode every original static-rate geometry independently, including all
    // image/frame headers and zero-duration intermediate layers. No golden
    // lengths and no estimate is used as the comparison's encoded cost.
    geometry.frames[2]
        .candidates
        .iter()
        .map(|candidate| {
            let crops: Vec<_> = candidate
                .rectangles
                .iter()
                .map(|&rect| crop(&images[2], rect).unwrap())
                .collect();
            let mut frames = vec![
                Frame {
                    image: &images[0],
                    options: FrameOptions {
                        duration: DURATIONS[0],
                        save_as_reference: 1,
                        ..Default::default()
                    },
                },
                Frame {
                    image: &images[1],
                    options: FrameOptions {
                        duration: DURATIONS[1],
                        save_as_reference: 2,
                        ..Default::default()
                    },
                },
            ];
            for (part, (rect, image)) in candidate.rectangles.iter().zip(&crops).enumerate() {
                let last = part + 1 == crops.len();
                frames.push(Frame {
                    image,
                    options: FrameOptions {
                        origin: (rect.x as i32, rect.y as i32),
                        duration: if last { DURATIONS[2] } else { 0 },
                        save_as_reference: if last { 0 } else { 2 },
                        blend: BlendInfo {
                            source: if part == 0 { candidate.source } else { 2 },
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                });
            }
            encode_frames(&images[0], &frames, options).unwrap()
        })
        .min_by_key(Vec::len)
        .unwrap()
}

#[test]
fn recent_full_rate_beats_independently_encoded_static_geometry() {
    let images = fixture();
    let options = options();
    let result = encode_auto(&images, &DURATIONS, &options, &auto(false)).unwrap();
    let anchor = result.frames[0]
        .candidates
        .iter()
        .find(|c| c.strategy == Strategy::Full)
        .unwrap()
        .encoded_bytes as f64
        / (SIDE * SIDE) as f64;
    let baseline = static_geometry_baseline(&images, &options, anchor);
    let implicit = AutoOptions {
        adaptive_splits: false,
        reference: Reference::Previous,
        allow_mask: false,
        tile_size: 16,
        max_rectangles: 16,
        prune: false,
        ..Default::default()
    };
    assert!(implicit.adaptive_rate_feedback);
    let default_result = encode_auto(&images, &DURATIONS, &options, &implicit).unwrap();
    assert_eq!(
        default_result.bytes, result.bytes,
        "feedback is the default partition policy"
    );
    let disabled = encode_auto(
        &images,
        &DURATIONS,
        &options,
        &AutoOptions {
            adaptive_rate_feedback: false,
            ..auto(false)
        },
    )
    .unwrap();
    assert_eq!(
        disabled.bytes.len(),
        baseline.len(),
        "legacy must preserve the independently encoded static geometry cost"
    );
    assert_eq!(
        disabled.bytes.len(),
        baseline.len(),
        "disabled feedback must retain static geometry costs"
    );
    assert!(disabled.frames.iter().all(|f| !f.adaptive_rate_feedback));
    assert!(result.frames.iter().all(|f| f.adaptive_rate_feedback));
    assert!(
        result.bytes.len() < baseline.len(),
        "feedback={}, static={}",
        result.bytes.len(),
        baseline.len()
    );
    let winner = &result.frames[2].candidates[result.frames[2].winner_index];
    assert_eq!(winner.strategy, Strategy::MergedPrevious);
    verify(&baseline, &images);
    verify(&result.bytes, &images);
    let pruned = encode_auto(&images, &DURATIONS, &options, &auto(true)).unwrap();
    assert_eq!(pruned.bytes, result.bytes);
    verify(&pruned.bytes, &images);
}

#[test]
fn feedback_keeps_hidden_float_and_wide_extra_words_and_explicit_limits() {
    let rgb = fixture();
    let hidden: Vec<_> = rgb
        .iter()
        .map(|image| {
            let PixelData::U8(values) = image.pixels() else {
                unreachable!()
            };
            let values = values
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 0])
                .collect();
            animated(
                Image::new(
                    SIDE as u32,
                    SIDE as u32,
                    PixelLayout::Rgba,
                    PixelData::U8(values),
                )
                .unwrap(),
                vec![],
            )
        })
        .collect();
    let floats: Vec<_> = rgb
        .iter()
        .map(|image| {
            let PixelData::U8(values) = image.pixels() else {
                unreachable!()
            };
            let words = [
                0,
                0x8000_0000,
                0x7fc1_2345,
                0x7f80_0000,
                0xff80_0000,
                1,
                0x3f00_0001,
            ];
            let values = values
                .chunks_exact(3)
                .map(|p| f32::from_bits(words[usize::from(p[0]) % words.len()]))
                .collect();
            animated(
                Image::new(
                    SIDE as u32,
                    SIDE as u32,
                    PixelLayout::Gray,
                    PixelData::F32(values),
                )
                .unwrap(),
                vec![],
            )
        })
        .collect();
    let wide: Vec<_> = rgb
        .iter()
        .map(|image| {
            let PixelData::U8(values) = image.pixels() else {
                unreachable!()
            };
            let main = values
                .chunks_exact(3)
                .map(|p| (u32::from(p[0]) << 23) | (u32::from(p[1]) << 8) | u32::from(p[2]))
                .collect();
            let extra = PixelData::U16(
                values
                    .chunks_exact(3)
                    .map(|p| u16::from(p[1]) * 257)
                    .collect(),
            );
            let image = Image::new(
                SIDE as u32,
                SIDE as u32,
                PixelLayout::Gray,
                PixelData::U32(main),
            )
            .unwrap();
            let mut metadata = image.metadata().clone();
            metadata.bit_depth = BitDepth::new(31).unwrap();
            metadata.animation = Some(Animation::default());
            metadata
                .extra_channels
                .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap());
            image.with_metadata(metadata, vec![extra]).unwrap()
        })
        .collect();
    for images in [&hidden, &floats, &wide] {
        let result = encode_auto(images, &DURATIONS, &options(), &auto(true)).unwrap();
        verify(&result.bytes, images);
    }
    for limited in [
        AutoOptions {
            max_total_trials: 1,
            ..auto(true)
        },
        AutoOptions {
            max_trial_pixels: (SIDE * SIDE - 1) as u64,
            ..auto(true)
        },
    ] {
        assert!(matches!(
            encode_auto(&rgb, &DURATIONS, &options(), &limited),
            Err(Error::Limit(_))
        ));
    }
}
