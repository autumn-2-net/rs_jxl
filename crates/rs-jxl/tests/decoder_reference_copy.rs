//! Exact background-copy regressions for saved, zero-duration crop layers.
//! The oracle composes the original samples directly, with independent color/alpha sources.
//! Small fixed-tool fixtures avoid MA training and external decoder dependencies.

use rs_jxl::decode::{DecodeOptions, DecodedFrame, SampleFormat, StreamDecoder, decode};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::frame::{BlendInfo, Frame, FrameOptions};
use rs_jxl::metadata::Orientation;
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode_frames};

const WIDTH: usize = 13;
const HEIGHT: usize = 9;

#[derive(Clone, Copy, Debug)]
enum Samples {
    U8,
    U16,
    F32,
}

impl Samples {
    fn pixels(self, words: Vec<u32>) -> PixelData {
        match self {
            Self::U8 => PixelData::U8(words.into_iter().map(|v| v as u8).collect()),
            Self::U16 => PixelData::U16(words.into_iter().map(|v| v as u16).collect()),
            Self::F32 => PixelData::F32(words.into_iter().map(f32::from_bits).collect()),
        }
    }

    fn word(self, i: usize, seed: usize) -> u32 {
        match self {
            Self::U8 => ((i * 53 + seed * 29 + 1) & 255) as u32,
            Self::U16 => ((i * 4051 + seed * 997 + 1) & 65535) as u32,
            // Finite dyadic floats, including negative zero, have an exact word oracle.
            Self::F32 => [-0.0f32, 0.03125, 0.25, 0.5, 0.75, 1.0][(i + seed) % 6].to_bits(),
        }
    }
}

fn encoding() -> EncodeOptions {
    EncodeOptions {
        threads: 1,
        max_output_bytes: 1 << 20,
        modular: ModularOptions {
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            rct: None,
            lz77: false,
            entropy: Coding::Prefix,
            tree_samples: 1,
            max_tree_leaves: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

struct Fixture {
    images: Vec<Image>,
    options: Vec<FrameOptions>,
    expected: Vec<u32>,
    layout: PixelLayout,
    orientation: Orientation,
}

fn replace(source: u8) -> BlendInfo {
    BlendInfo {
        source,
        ..Default::default()
    }
}

fn fixture(samples: Samples, layout: PixelLayout, orientation: Orientation) -> Fixture {
    reference_fixture(samples, layout, orientation, false)
}

fn reference_fixture(
    samples: Samples,
    layout: PixelLayout,
    orientation: Orientation,
    same_slot: bool,
) -> Fixture {
    // Empty slot 3 supplies positive zero to layer 1, despite a nonzero full reference in 1.
    // Layers 2 and 4 save into a different slot; layer 3 overwrites its own color source.
    // Independent alpha sources in layers 2/3 must also apply *outside* their small crops.
    let mut specs = [
        (WIDTH, HEIGHT, (0, 0), 0, 0, 1),
        (5, 4, (-2, 1), 3, 3, 2),
        (5, 4, (11, 7), 2, 1, 3),
        (3, 2, (4, 3), 3, 2, 3),
        (2, 3, (8, -1), 3, 3, 2),
        (4, 4, (1, 7), 2, 2, 0),
    ];
    if same_slot {
        let last = specs.len() - 1;
        for (index, (_, _, _, color, alpha, destination)) in specs.iter_mut().enumerate() {
            *color = u8::from(index != 0);
            *alpha = *color;
            *destination = u8::from(index != last);
        }
    }
    let channels = layout.channels();
    let mut images = vec![];
    let mut options = vec![];
    let mut slots: [Option<Vec<u32>>; 4] = [None, None, None, None];
    let mut expected = vec![];
    for (index, &(width, height, origin, color_source, alpha_source, destination)) in
        specs.iter().enumerate()
    {
        let mut words: Vec<_> = (0..width * height * channels)
            .map(|i| samples.word(i, index + 1))
            .collect();
        if layout.has_alpha() {
            for (i, pixel) in words.chunks_exact_mut(channels).enumerate() {
                if i % 3 == 0 {
                    pixel[3] = 0;
                    // Explicitly retain nonzero hidden color in every crop.
                    pixel[0] = samples.word(1, 0);
                }
            }
        }
        let image = Image::new(
            width as u32,
            height as u32,
            layout,
            samples.pixels(words.clone()),
        )
        .unwrap();
        let mut metadata = image.metadata().clone();
        metadata.orientation = orientation;
        metadata.modular_16_bit_buffer_sufficient = matches!(samples, Samples::U8);
        images.push(image.with_metadata(metadata, vec![]).unwrap());
        options.push(FrameOptions {
            origin,
            blend: replace(color_source),
            extra_blend: if layout.has_alpha() {
                vec![replace(alpha_source)]
            } else {
                vec![]
            },
            save_as_reference: destination,
            name: format!("reference-copy-{index}"),
            ..Default::default()
        });

        // Independent sample-domain Replace compositor, including clipping of negative and
        // bottom/right origins. No decoded frame or alternate thread count is the oracle.
        expected = vec![0; WIDTH * HEIGHT * channels];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let crop_x = x as i32 - origin.0;
                let crop_y = y as i32 - origin.1;
                for c in 0..channels {
                    let output = (y * WIDTH + x) * channels + c;
                    let source = if c < 3 { color_source } else { alpha_source };
                    if let Some(bg) = &slots[source as usize] {
                        expected[output] = bg[output];
                    }
                    if crop_x >= 0
                        && crop_y >= 0
                        && (crop_x as usize) < width
                        && (crop_y as usize) < height
                    {
                        expected[output] =
                            words[(crop_y as usize * width + crop_x as usize) * channels + c];
                    }
                }
            }
        }
        if index + 1 != specs.len() {
            slots[destination as usize] = Some(expected.clone());
        }
    }
    Fixture {
        images,
        options,
        expected,
        layout,
        orientation,
    }
}

fn encode_fixture(fixture: &Fixture) -> Vec<u8> {
    let frames: Vec<_> = fixture
        .images
        .iter()
        .zip(&fixture.options)
        .map(|(image, options)| Frame {
            image,
            options: options.clone(),
        })
        .collect();
    encode_frames(&fixture.images[0], &frames, &encoding()).unwrap()
}

fn words(pixels: &PixelData) -> Vec<u32> {
    match pixels {
        PixelData::U8(v) => v.iter().map(|&v| u32::from(v)).collect(),
        PixelData::U16(v) => v.iter().map(|&v| u32::from(v)).collect(),
        PixelData::F32(v) => v.iter().map(|v| v.to_bits()).collect(),
        other => panic!("unexpected output sample type: {:?}", other.bits()),
    }
}

fn assert_words(actual: &PixelData, expected: &[u32], context: &str) {
    let actual = words(actual);
    assert_eq!(actual.len(), expected.len(), "{context}");
    if let Some(i) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{context}: sample {i}: actual={:#010x}, expected={:#010x}",
            actual[i], expected[i]
        );
    }
}

fn orient(
    input: &[u32],
    width: usize,
    height: usize,
    channels: usize,
    orientation: Orientation,
) -> ((usize, usize), Vec<u32>) {
    if orientation == Orientation::Identity {
        return ((width, height), input.to_vec());
    }
    assert_eq!(orientation, Orientation::Rotate90Clockwise);
    let mut output = vec![0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let src = (y * width + x) * channels;
            let dst = (x * height + height - 1 - y) * channels;
            output[dst..dst + channels].copy_from_slice(&input[src..src + channels]);
        }
    }
    ((height, width), output)
}

fn assert_frame(
    frame: &DecodedFrame,
    expected: &[u32],
    size: (usize, usize),
    fixture: &Fixture,
    context: &str,
) {
    let ((width, height), expected) = orient(
        expected,
        size.0,
        size.1,
        fixture.layout.channels(),
        fixture.orientation,
    );
    assert!(frame.complete, "{context}");
    assert_eq!((frame.width, frame.height), (width, height), "{context}");
    assert_eq!(frame.layout, fixture.layout, "{context}");
    assert_words(&frame.pixels, &expected, context);
    if fixture.layout.has_alpha() {
        let alpha: Vec<_> = expected.chunks_exact(4).map(|p| p[3]).collect();
        assert_eq!(frame.extra_planes.len(), 1, "{context}");
        assert_words(&frame.extra_planes[0], &alpha, context);
    } else {
        assert!(frame.extra_planes.is_empty(), "{context}");
    }
}

fn assert_coalesced(frames: &[DecodedFrame], fixture: &Fixture, context: &str) {
    assert_eq!(frames.len(), 1, "{context}");
    assert_eq!(frames[0].index, 0, "{context}");
    assert_frame(
        &frames[0],
        &fixture.expected,
        (WIDTH, HEIGHT),
        fixture,
        context,
    );
}

fn assert_incremental(bytes: &[u8], options: &DecodeOptions, fixture: &Fixture, context: &str) {
    let mut decoder = StreamDecoder::new(options).unwrap();
    let mut frames = vec![];
    decoder.push(&[]).unwrap();
    for chunk in bytes.chunks(7) {
        decoder.push(chunk).unwrap();
        frames.extend(decoder.take_frames());
    }
    let summary = decoder.finish().unwrap();
    frames.extend(decoder.take_frames());
    assert!(summary.complete, "{context}");
    assert_eq!(summary.frame_count, 1, "{context}");
    assert_coalesced(&frames, fixture, context);
}

#[test]
fn chained_reference_crops_preserve_rgb_rgba8_16_and_hidden_color() {
    for samples in [Samples::U8, Samples::U16] {
        for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
            let fixture = fixture(samples, layout, Orientation::Identity);
            assert!(
                fixture
                    .expected
                    .chunks_exact(layout.channels())
                    .any(|p| { p[..3].iter().all(|&v| v == 0) })
            );
            if layout.has_alpha() {
                assert!(
                    fixture
                        .expected
                        .chunks_exact(4)
                        .any(|p| { p[3] == 0 && p[..3].iter().any(|&v| v != 0) })
                );
            }
            let bytes = encode_fixture(&fixture);
            for threads in [1, 4] {
                let options = DecodeOptions {
                    threads,
                    ..Default::default()
                };
                let context = format!("{samples:?}/{layout:?}, {threads} threads");
                let decoded = decode(&bytes, &options).unwrap();
                assert!(decoded.complete, "{context}");
                assert_coalesced(&decoded.frames, &fixture, &context);
                assert_incremental(&bytes, &options, &fixture, &context);
            }
        }
    }
}

#[test]
fn finite_f32_words_and_oriented_output_keep_saved_background_exact() {
    for (samples, layout, orientation) in [
        (Samples::F32, PixelLayout::Rgba, Orientation::Identity),
        (
            Samples::U16,
            PixelLayout::Rgba,
            Orientation::Rotate90Clockwise,
        ),
    ] {
        let fixture = fixture(samples, layout, orientation);
        let bytes = encode_fixture(&fixture);
        let options = DecodeOptions {
            threads: 4,
            format: if matches!(samples, Samples::F32) {
                SampleFormat::F32
            } else {
                SampleFormat::Native
            },
            ..Default::default()
        };
        let context = format!("{samples:?}/{orientation:?} output fallback");
        let decoded = decode(&bytes, &options).unwrap();
        assert!(decoded.complete, "{context}");
        assert_coalesced(&decoded.frames, &fixture, &context);
        assert_incremental(&bytes, &options, &fixture, &context);
    }
}

#[test]
fn noncoalescing_returns_original_crop_samples_after_reference_updates() {
    let fixture = fixture(Samples::U8, PixelLayout::Rgba, Orientation::Identity);
    let bytes = encode_fixture(&fixture);
    let options = DecodeOptions {
        threads: 4,
        coalescing: false,
        ..Default::default()
    };
    let decoded = decode(&bytes, &options).unwrap();
    assert!(decoded.complete);
    assert_eq!(decoded.frames.len(), fixture.images.len());
    for (index, (frame, image)) in decoded.frames.iter().zip(&fixture.images).enumerate() {
        assert_eq!(frame.index, index);
        assert_frame(
            frame,
            &words(image.pixels()),
            (image.width() as usize, image.height() as usize),
            &fixture,
            "raw layer fallback",
        );
    }
}

#[test]
fn hidden_replace_layers_update_one_reference_canvas_without_losing_samples() {
    for samples in [Samples::U8, Samples::U16, Samples::F32] {
        for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
            let fixture = reference_fixture(samples, layout, Orientation::Identity, true);
            let bytes = encode_fixture(&fixture);
            for threads in [1, 4] {
                let options = DecodeOptions {
                    threads,
                    ..Default::default()
                };
                let context = format!("same-slot {samples:?}/{layout:?}/{threads}");
                let decoded = decode(&bytes, &options).unwrap();
                assert!(decoded.complete);
                assert_coalesced(&decoded.frames, &fixture, &context);
                assert_incremental(&bytes, &options, &fixture, &context);
            }
            let raw = decode(
                &bytes,
                &DecodeOptions {
                    coalescing: false,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(raw.complete);
            assert_eq!(raw.frames.len(), fixture.images.len());
            for (frame, image) in raw.frames.iter().zip(&fixture.images) {
                assert_frame(
                    frame,
                    &words(image.pixels()),
                    (image.width() as usize, image.height() as usize),
                    &fixture,
                    "same-slot raw layer fallback",
                );
            }
        }
    }
}

#[test]
fn flushing_hidden_reference_layers_leaves_api_pixels_until_visible_frame() {
    use jxl_decoder::api::{
        Event, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat,
    };

    for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
        let fixture = reference_fixture(Samples::U8, layout, Orientation::Identity, true);
        let bytes = encode_fixture(&fixture);
        let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());
        let mut pixels = vec![0xa5; WIDTH * HEIGHT * layout.channels()];
        let mut pending = vec![];
        let mut supplied = 0;
        let mut basic_info_seen = false;
        let mut visible_header_seen = false;
        let mut hidden_flushes = 0;
        let mut events = vec![];
        loop {
            let (event, consumed) = {
                let mut input = pending.as_slice();
                let mut buffers = [JxlOutputBuffer::new(
                    &mut pixels,
                    HEIGHT,
                    WIDTH * layout.channels(),
                )];
                let event = decoder
                    .process(
                        &mut input,
                        visible_header_seen.then_some(&mut buffers[..]),
                        None,
                    )
                    .unwrap();
                (event, pending.len() - input.len())
            };
            // Preserve unconsumed bytes when a header or section straddles the next chunk.
            pending.drain(..consumed);
            match event {
                Event::BasicInfo => {
                    basic_info_seen = true;
                    events.push("BasicInfo");
                    decoder
                        .set_pixel_format(JxlPixelFormat {
                            color_type: if layout.has_alpha() {
                                JxlColorType::Rgba
                            } else {
                                JxlColorType::Rgb
                            },
                            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
                            // Request alpha only in the main RGBA buffer, keeping one API buffer.
                            extra_channel_format: vec![None; usize::from(layout.has_alpha())],
                        })
                        .unwrap();
                }
                Event::NeedMoreInput { .. } => {
                    if basic_info_seen && !visible_header_seen {
                        assert!(decoder.frame_header().is_none());
                        let mut buffers = [JxlOutputBuffer::new(
                            &mut pixels,
                            HEIGHT,
                            WIDTH * layout.channels(),
                        )];
                        // The dirty return value includes internal reference activity. Only
                        // externally observable pixels and visible frame events are asserted.
                        let _ = decoder.flush_pixels(&mut buffers, None).unwrap();
                        assert!(pixels.iter().all(|&v| v == 0xa5), "{layout:?}");
                        hidden_flushes += 1;
                    }
                    let end = (supplied + 7).min(bytes.len());
                    assert!(
                        end > supplied,
                        "complete {layout:?} fixture needs more input"
                    );
                    pending.extend_from_slice(&bytes[supplied..end]);
                    supplied = end;
                }
                Event::FrameHeader => {
                    assert!(!visible_header_seen);
                    visible_header_seen = true;
                    events.push("FrameHeader");
                    let header = decoder.frame_header().unwrap();
                    assert_eq!(header.size, (WIDTH, HEIGHT));
                    assert_eq!(header.name, fixture.options.last().unwrap().name);
                    assert!(header.is_last);
                }
                Event::FrameComplete { has_more_frames } => {
                    assert!(!has_more_frames);
                    events.push("FrameComplete");
                    assert_words(
                        &PixelData::U8(pixels.clone()),
                        &fixture.expected,
                        "visible output after hidden flushes",
                    );
                }
                Event::Complete => {
                    events.push("Complete");
                    break;
                }
            }
        }
        assert!(hidden_flushes > 0);
        assert_eq!(
            events,
            ["BasicInfo", "FrameHeader", "FrameComplete", "Complete"]
        );
    }
}
