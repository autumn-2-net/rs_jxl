//! Small multigroup scheduling regressions with original samples as the oracle.
//! Fixed-tool fixtures, with tiny typed-storage and retained-frame regressions;
//! no MA training, tool search, LZ77 search or external decoder processes.

use rs_jxl::decode::{DecodeOptions, DecodedFrame, DecodedImage, StreamDecoder, decode};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::metadata::{Animation, BitDepth, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::multiframe::{self, Reference};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode, encode_frames};

const WIDTH: u32 = 131;
const HEIGHT: u32 = 133;

#[derive(Clone, Copy, Debug)]
enum Route {
    Plain,
    GradientRct,
    WeightedRct,
    Palette,
    Squeeze,
}

fn encoding(route: Route) -> EncodeOptions {
    let mut options = EncodeOptions {
        threads: 1,
        max_pixels: u64::from(WIDTH) * u64::from(HEIGHT),
        max_output_bytes: 2 << 20,
        // Four groups: 128x128, 3x128, 128x5 and 3x5.
        group_size_shift: 0,
        modular: ModularOptions {
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            rct: None,
            entropy: Coding::Prefix,
            lz77: false,
            tree_samples: 1,
            max_tree_leaves: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    match route {
        Route::Plain => {}
        Route::GradientRct => {
            options.modular.predictor = Predictor::Gradient;
            options.modular.fixed_tree = Some(FixedTree::GradientLeaf);
            options.modular.rct = Some(41);
        }
        Route::WeightedRct => {
            options.modular.predictor = Predictor::Weighted;
            options.modular.fixed_tree = Some(FixedTree::Weighted);
            options.modular.rct = Some(6);
        }
        Route::Palette => {
            // The fixture has only eight RGBA tuples, below this explicit limit.
            options.modular.palette_colors = Some(16);
        }
        Route::Squeeze => options.modular.squeeze = vec![true, false],
    }
    options
}

fn source(bits: u8, width: u32, height: u32) -> Image {
    let colors: [[u16; 4]; 8] = if bits == 8 {
        [
            [1, 255, 129, 0],
            [255, 128, 3, 255],
            [17, 0, 113, 128],
            [91, 243, 52, 0],
            [0, 1, 255, 1],
            [254, 127, 129, 254],
            [85, 170, 34, 64],
            [255, 255, 255, 255],
        ]
    } else {
        [
            [1, 65535, 32769, 0],
            [65535, 32768, 257, 65535],
            [4097, 0, 32767, 32768],
            [23457, 62465, 13313, 0],
            [0, 1, 65535, 1],
            [65534, 32767, 32769, 65534],
            [21847, 43691, 8739, 16385],
            [65535, 65535, 65535, 65535],
        ]
    };
    let mut samples = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height as usize {
        for x in 0..width as usize {
            let index = ((x * 3 + y * 5) ^ (x >> 3) ^ (y >> 2)) & 7;
            samples.extend_from_slice(&colors[index]);
        }
    }
    assert!(
        samples
            .chunks_exact(4)
            .any(|p| p[3] == 0 && p[..3].iter().any(|&v| v != 0))
    );
    let pixels = if bits == 8 {
        PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
    } else {
        PixelData::U16(samples)
    };
    let image = Image::new(width, height, PixelLayout::Rgba, pixels).unwrap();
    let mut metadata = image.metadata().clone();
    // Cover the backend's I16 and I32 Modular storage without wide capture.
    metadata.modular_16_bit_buffer_sufficient = bits == 8;
    image.with_metadata(metadata, vec![]).unwrap()
}

fn alpha(source: &Image) -> PixelData {
    match source.pixels() {
        PixelData::U8(v) => PixelData::U8(v.chunks_exact(4).map(|p| p[3]).collect()),
        PixelData::U16(v) => PixelData::U16(v.chunks_exact(4).map(|p| p[3]).collect()),
        _ => unreachable!(),
    }
}

fn assert_original(frame: &DecodedFrame, source: &Image, alpha: &PixelData, context: &str) {
    assert!(frame.complete, "{context}");
    assert_eq!(frame.index, 0, "{context}");
    assert_eq!(
        (frame.width, frame.height),
        (source.width() as usize, source.height() as usize),
        "{context}"
    );
    assert_eq!(frame.layout, PixelLayout::Rgba, "{context}");
    // Includes invisible RGB; neither compositing nor a 1T baseline is the oracle.
    assert_eq!(&frame.pixels, source.pixels(), "{context}");
    assert_eq!(
        frame.extra_planes.as_slice(),
        std::slice::from_ref(alpha),
        "{context}"
    );
}

fn assert_incremental(
    bytes: &[u8],
    options: &DecodeOptions,
    batch: &DecodedImage,
    source: &Image,
    alpha: &PixelData,
    context: &str,
) {
    let mut stream = StreamDecoder::new(options).unwrap();
    stream.push(&bytes[..1]).unwrap();
    stream.push(&[]).unwrap();
    let mut frames = stream.take_frames();
    // Non-power-of-two chunks split headers and group payloads without bytewise replay.
    for chunk in bytes[1..].chunks(4093) {
        stream.push(chunk).unwrap();
        frames.extend(stream.take_frames());
    }
    let summary = stream.finish().unwrap();
    frames.extend(stream.take_frames());
    assert!(summary.complete, "{context}");
    assert!(!summary.has_preview, "{context}");
    assert_eq!(summary.frame_count, 1, "{context}");
    assert_eq!(summary.info.size, batch.info.size, "{context}");
    assert_eq!(summary.info.bit_depth, batch.info.bit_depth, "{context}");
    assert_eq!(frames.len(), 1, "{context}");
    assert_original(&frames[0], source, alpha, context);
    assert_eq!(frames[0].pixels, batch.frames[0].pixels, "{context}");
    assert_eq!(
        frames[0].extra_planes, batch.frames[0].extra_planes,
        "{context}"
    );
    assert_eq!(
        format!("{:?}", frames[0].header),
        format!("{:?}", batch.frames[0].header),
        "{context}",
    );
}

fn roundtrip(route: Route) {
    for bits in [8, 16] {
        let source = source(bits, WIDTH, HEIGHT);
        let expected_alpha = alpha(&source);
        let options = encoding(route);
        let group_dim = 128usize << options.group_size_shift;
        assert_eq!(
            (WIDTH as usize).div_ceil(group_dim) * (HEIGHT as usize).div_ceil(group_dim),
            4
        );
        // Encode once, reusing the identical codestream for all worker counts.
        let bytes = encode(&source, &options).unwrap();
        for threads in [1, 2, 8] {
            let options = DecodeOptions {
                threads,
                ..Default::default()
            };
            let context = format!("{route:?}, U{bits}, threads={threads}");
            let batch = decode(&bytes, &options).unwrap();
            assert!(batch.complete, "{context}");
            assert!(batch.preview.is_none(), "{context}");
            assert_eq!(
                batch.info.bit_depth.bits_per_sample(),
                u32::from(bits),
                "{context}"
            );
            assert_eq!(batch.frames.len(), 1, "{context}");
            assert_original(&batch.frames[0], &source, &expected_alpha, &context);
            if matches!(route, Route::GradientRct)
                || (threads == 8 && matches!(route, Route::Palette))
            {
                assert_incremental(&bytes, &options, &batch, &source, &expected_alpha, &context);
            }
        }
    }
}

#[test]
fn multigroup_rct_gradient_and_weighted_preserve_u8_u16_hidden_rgb() {
    for route in [Route::GradientRct, Route::WeightedRct] {
        roundtrip(route);
    }
}

#[test]
fn multigroup_palette_and_squeeze_preserve_u8_u16_hidden_rgb() {
    for route in [Route::Palette, Route::Squeeze] {
        roundtrip(route);
    }
}

#[test]
fn full_decode_large_rgba8_typed_destination_is_exact() {
    let source = source(8, 513, 513);
    let expected_alpha = alpha(&source);
    let PixelData::U8(samples) = source.pixels() else {
        unreachable!()
    };
    // Keep the large odd-sized fixture previously used for the 1 MiB copy gate.
    assert!(samples.len() > 1024 * 1024);
    assert!(expected_alpha.len() < 1024 * 1024);
    let mut options = encoding(Route::Plain);
    options.max_pixels = u64::from(source.width()) * u64::from(source.height());
    options.max_output_bytes = 4 << 20;
    options.group_size_shift = 1;
    let bytes = encode(&source, &options).unwrap();
    for threads in [1, 2, 8] {
        let options = DecodeOptions {
            threads,
            ..Default::default()
        };
        let context = format!("513x513 RGBA8, threads={threads}");
        let decoded = decode(&bytes, &options).unwrap();
        assert!(decoded.complete, "{context}");
        assert_eq!(decoded.frames.len(), 1, "{context}");
        assert_original(&decoded.frames[0], &source, &expected_alpha, &context);
    }
}

#[test]
fn typed_main_and_mixed_extras_preserve_float_words() {
    const WORDS: [u32; 12] = [
        0,
        0x8000_0000,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0x7fc0_0123,
        0xff80_0123,
        0xffff_ffff,
    ];
    let count = 17 * 3;
    let extra_words: Vec<_> = (0..count).map(|i| WORDS[i % WORDS.len()]).collect();
    let extras = vec![
        PixelData::U8((0..count).map(|i| (i * 37) as u8).collect()),
        PixelData::U16((0..count).map(|i| (i * 8191) as u16).collect()),
        PixelData::F32(extra_words.iter().copied().map(f32::from_bits).collect()),
    ];
    for bits in [8, 16, 32] {
        let pixels = match bits {
            8 => PixelData::U8((0..count * 3).map(|i| (i * 37) as u8).collect()),
            16 => PixelData::U16((0..count * 3).map(|i| (i * 8191) as u16).collect()),
            _ => PixelData::F32(
                (0..count * 3)
                    .map(|i| f32::from_bits(WORDS[i % WORDS.len()]))
                    .collect(),
            ),
        };
        let image = Image::new(17, 3, PixelLayout::Rgb, pixels).unwrap();
        let mut metadata = image.metadata().clone();
        for (kind, depth, name) in [
            (
                ExtraChannelType::SelectionMask,
                BitDepth::new(8).unwrap(),
                "selection",
            ),
            (ExtraChannelType::Depth, BitDepth::new(16).unwrap(), "depth"),
            (
                ExtraChannelType::Optional,
                BitDepth::floating(32, 8).unwrap(),
                "raw\0float",
            ),
        ] {
            let mut info = ExtraChannelInfo::new(kind, 8).unwrap();
            info.bit_depth = depth;
            info.name = name.into();
            metadata.extra_channels.push(info);
        }
        let source = image.with_metadata(metadata, extras.clone()).unwrap();
        let bytes = encode(&source, &encoding(Route::Plain)).unwrap();
        for threads in [1, 2, 8] {
            let options = DecodeOptions {
                threads,
                ..Default::default()
            };
            let batch = decode(&bytes, &options).unwrap();
            let mut stream = StreamDecoder::new(&options).unwrap();
            for chunk in bytes.chunks(31) {
                stream.push(chunk).unwrap();
            }
            assert!(stream.finish().unwrap().complete);
            let streamed = stream.take_frames();
            assert!(batch.complete);
            assert_eq!(batch.frames.len(), 1);
            assert_eq!(streamed.len(), 1);
            for frame in [&batch.frames[0], &streamed[0]] {
                assert!(frame.complete);
                assert_eq!(frame.layout, PixelLayout::Rgb);
                assert_eq!((frame.width, frame.height), (17, 3));
                assert_eq!(&frame.pixels, source.pixels());
                assert_eq!(frame.extra_planes, extras);
                let PixelData::F32(values) = &frame.extra_planes[2] else {
                    panic!("expected F32 extra plane");
                };
                assert_eq!(
                    values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    extra_words
                );
                if let PixelData::F32(values) = &frame.pixels {
                    let expected: Vec<_> = (0..count * 3).map(|i| WORDS[i % WORDS.len()]).collect();
                    assert_eq!(
                        values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected
                    );
                }
            }
            for (actual, expected) in batch
                .info
                .extra_channels
                .iter()
                .zip(&source.metadata().extra_channels)
            {
                assert_eq!(actual.name, expected.name);
                assert_eq!(
                    actual.bit_depth.bits_per_sample(),
                    u32::from(expected.bit_depth.bits_per_sample)
                );
            }
        }
    }
}

#[test]
fn completed_frames_remain_owned_while_stream_decodes_next_frame() {
    let first = source(16, 17, 5);
    let mut metadata = first.metadata().clone();
    metadata.animation = Some(Animation {
        ticks_per_second_numerator: 60,
        ..Default::default()
    });
    let PixelData::U16(mut pixels) = first.pixels().clone() else {
        unreachable!()
    };
    for pixel in pixels.chunks_exact_mut(4) {
        pixel[0] ^= 0x5a5a;
        pixel[1] ^= 0xa5a5;
    }
    let second = Image::new(17, 5, PixelLayout::Rgba, PixelData::U16(pixels)).unwrap();
    let images = [
        first.with_metadata(metadata.clone(), vec![]).unwrap(),
        second.with_metadata(metadata, vec![]).unwrap(),
    ];
    let durations = [2, 3];
    let plan = multiframe::plan(&images, &durations, Reference::Full).unwrap();
    let frames: Vec<_> = plan.iter().map(|frame| frame.as_frame()).collect();
    let bytes = encode_frames(&images[0], &frames, &encoding(Route::Plain)).unwrap();
    let check = |frames: &[DecodedFrame]| {
        for (i, frame) in frames.iter().enumerate() {
            assert!(frame.complete);
            assert_eq!(frame.index, i);
            assert_eq!(frame.duration_ticks, durations[i]);
            assert_eq!(&frame.pixels, images[i].pixels());
            assert_eq!(frame.extra_planes, vec![alpha(&images[i])]);
        }
    };
    for threads in [1, 2, 8] {
        let options = DecodeOptions {
            threads,
            ..Default::default()
        };
        let batch = decode(&bytes, &options).unwrap();
        assert_eq!(batch.frames.len(), 2);
        check(&batch.frames);
        let mut stream = StreamDecoder::new(&options).unwrap();
        let mut held = Vec::new();
        let mut held_before_end = false;
        for chunk in bytes.chunks(13) {
            stream.push(chunk).unwrap();
            held.extend(stream.take_frames());
            held_before_end |= held.len() == 1;
            check(&held);
        }
        assert!(stream.finish().unwrap().complete);
        held.extend(stream.take_frames());
        drop(stream);
        assert!(held_before_end);
        assert_eq!(held.len(), 2);
        check(&held);
    }
}
