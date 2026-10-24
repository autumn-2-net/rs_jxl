//! Exact sample and layer geometry contracts, independent of lossy rendering.

use rs_jxl::decode::{DecodeOptions, decode, decode_frame, inspect};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::frame::{BlendInfo, Frame, FrameOptions};
use rs_jxl::metadata::{BitDepth, Orientation};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::{
    EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout, encode, encode_frames,
};

fn options() -> EncodeOptions {
    EncodeOptions {
        modular: ModularOptions {
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            rct: None,
            lz77: false,
            entropy: Coding::Prefix,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn every_native_integer_value_through_sixteen_bits_roundtrips() {
    for bits in 1..=16 {
        let count = 1usize << bits;
        let width = count.min(256);
        let pixels = if bits <= 8 {
            PixelData::U8((0..count).map(|x| (x as u8).wrapping_mul(197)).collect())
        } else {
            PixelData::U16(
                (0..count)
                    .map(|x| ((x * 40503) & (count - 1)) as u16)
                    .collect(),
            )
        };
        // Keep low-bit samples within their declared range as well.
        let pixels = match pixels {
            PixelData::U8(values) => {
                PixelData::U8(values.into_iter().map(|x| x & (count - 1) as u8).collect())
            }
            other => other,
        };
        let image = Image::new(
            width as u32,
            (count / width) as u32,
            PixelLayout::Gray,
            pixels,
        )
        .unwrap();
        let mut metadata = image.metadata().clone();
        metadata.bit_depth = BitDepth::new(bits).unwrap();
        let image = image.with_metadata(metadata, vec![]).unwrap();
        let bytes = encode(&image, &options()).unwrap();
        for threads in [1, 4] {
            let decoded = decode(
                &bytes,
                &DecodeOptions {
                    threads,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(decoded.complete);
            assert_eq!(decoded.info.bit_depth.bits_per_sample(), u32::from(bits));
            assert_eq!(
                decoded.frames[0].pixels,
                *image.pixels(),
                "{bits} bits, {threads} threads"
            );
        }
    }
}

#[test]
fn backend_sample_limits_are_not_reported_as_corrupt_pixels() {
    let image = Image::new(4, 3, PixelLayout::Rgb, PixelData::U8(vec![127; 36])).unwrap();
    let bytes = encode(&image, &options()).unwrap();
    let mut options = DecodeOptions::default();
    options.limits.max_samples = 1;
    assert!(matches!(decode(&bytes, &options), Err(Error::Limit(_))));
    assert!(matches!(
        decode_frame(&bytes, 0, &options),
        Err(Error::Limit(_))
    ));
    assert!(matches!(inspect(&bytes, &options), Err(Error::Limit(_))));
    assert!(matches!(
        rs_jxl::decode::read_basic_info(&bytes, &options),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        decode(b"not a JXL", &options),
        Err(Error::Decoding(_))
    ));
}

// Map individual pixel coordinates, including coordinates outside the canvas.
// Expected layer bounds are derived from mapped pixels, not the decoder formula.
fn orient(x: i64, y: i64, width: i64, height: i64, orientation: u8) -> (i64, i64) {
    match orientation {
        1 => (x, y),
        2 => (width - 1 - x, y),
        3 => (width - 1 - x, height - 1 - y),
        4 => (x, height - 1 - y),
        5 => (y, x),
        6 => (height - 1 - y, x),
        7 => (height - 1 - y, width - 1 - x),
        8 => (y, width - 1 - x),
        _ => unreachable!(),
    }
}

fn oriented(values: &[u8], width: usize, height: usize, orientation: u8) -> Vec<u8> {
    let output_width = if orientation >= 5 { height } else { width };
    let mut output = vec![0; values.len()];
    for (i, &value) in values.iter().enumerate() {
        let (x, y) = orient(
            (i % width) as i64,
            (i / width) as i64,
            width as i64,
            height as i64,
            orientation,
        );
        output[y as usize * output_width + x as usize] = value;
    }
    output
}

#[test]
fn oriented_off_canvas_layers_preserve_samples_and_reconstruct_exactly() {
    let background: Vec<u8> = (0..15).map(|i| i * 7).collect();
    let layer = vec![201, 202, 203, 204, 205, 206];
    for orientation in 1..=8 {
        let canvas =
            Image::new(5, 3, PixelLayout::Gray, PixelData::U8(background.clone())).unwrap();
        let mut metadata = canvas.metadata().clone();
        metadata.orientation = Orientation::try_from(orientation).unwrap();
        let canvas = canvas.with_metadata(metadata.clone(), vec![]).unwrap();
        let patch = Image::new(3, 2, PixelLayout::Gray, PixelData::U8(layer.clone()))
            .unwrap()
            .with_metadata(metadata, vec![])
            .unwrap();
        for origin in [(-1, 1), (3, -1), (1, 2)] {
            let frames = [
                Frame {
                    image: &canvas,
                    options: FrameOptions {
                        save_as_reference: 1,
                        ..Default::default()
                    },
                },
                Frame {
                    image: &patch,
                    options: FrameOptions {
                        origin,
                        blend: BlendInfo {
                            source: 1,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                },
            ];
            let bytes = encode_frames(&canvas, &frames, &options()).unwrap();
            let mut composite = background.clone();
            for (i, &value) in layer.iter().enumerate() {
                let x = origin.0 + (i % 3) as i32;
                let y = origin.1 + (i / 3) as i32;
                if (0..5).contains(&x) && (0..3).contains(&y) {
                    composite[y as usize * 5 + x as usize] = value;
                }
            }
            for adjust_orientation in [false, true] {
                let applied = if adjust_orientation { orientation } else { 1 };
                let options = DecodeOptions {
                    adjust_orientation,
                    ..Default::default()
                };
                let decoded = decode(&bytes, &options).unwrap();
                assert_eq!(decoded.frames.len(), 1);
                assert_eq!(
                    decoded.frames[0].pixels,
                    PixelData::U8(oriented(&composite, 5, 3, applied))
                );
                let raw_options = DecodeOptions {
                    coalescing: false,
                    ..options
                };
                let raw = decode(&bytes, &raw_options).unwrap();
                assert_eq!(raw.frames.len(), 2);
                let expected_pixels = PixelData::U8(oriented(&layer, 3, 2, applied));
                assert_eq!(raw.frames[1].pixels, expected_pixels);
                let positions: Vec<_> = (0..6)
                    .map(|i| {
                        orient(
                            i64::from(origin.0) + i % 3,
                            i64::from(origin.1) + i / 3,
                            5,
                            3,
                            applied,
                        )
                    })
                    .collect();
                let expected_origin = (
                    positions.iter().map(|p| p.0).min().unwrap(),
                    positions.iter().map(|p| p.1).min().unwrap(),
                );
                assert_eq!(raw.frames[1].header.layer_info.crop_origin, expected_origin);
                assert_eq!(raw.frames[1].duration_ticks, 0);
                let selected = decode_frame(&bytes, 1, &raw_options).unwrap();
                assert_eq!(selected.frames[0].pixels, expected_pixels);
                let scanned = inspect(&bytes, &raw_options).unwrap();
                assert_eq!(scanned.frames[1].layer_info.crop_origin, expected_origin);
                assert_eq!(scanned.frames[1].size, raw.frames[1].header.size);
            }
        }
    }
}
