//! Bounded interaction coverage, not an effort/tool acceptance matrix.
//! 24 tiny encodes per matrix; decoding tests both thread counts independently.
//! All expected displayed states originate here, not from decoded Rust output.

use super::*;
use rs_jxl::decode::{DecodeOptions, DecodedFrame, Decoder};

fn integer_pixels(bits: u8, values: Vec<u16>) -> PixelData {
    match bits {
        8 => PixelData::U8(
            values
                .into_iter()
                .map(|v| u8::try_from(v).unwrap())
                .collect(),
        ),
        16 => PixelData::U16(values),
        _ => panic!("unsupported combination fixture depth"),
    }
}

fn combination_images(bits: u8, timing: Timing) -> Vec<Image> {
    let maximum = if bits == 8 { 255 } else { 65535 };
    let mut rgba: Vec<u16> = (0..15 * 4)
        .map(|i| ((i * 1091 + 17) & maximum) as u16)
        .collect();
    for (i, pixel) in rgba.chunks_exact_mut(4).enumerate() {
        pixel[3] = [0, 1, maximum / 2, maximum - 1, maximum][i % 5] as u16;
    }
    let mut depth: Vec<u16> = (0..15).map(|i| ((i * 2137 + 3) & maximum) as u16).collect();
    let make = |rgba: Vec<u16>, depth: Vec<u16>| {
        let base = Image::new(5, 3, PixelLayout::Rgba, integer_pixels(bits, rgba)).unwrap();
        let mut metadata = base.metadata().clone();
        if !matches!(timing, Timing::Layers) {
            metadata.animation = Some(Animation::default());
        }
        metadata
            .extra_channels
            .push(ExtraChannelInfo::new(ExtraChannelType::Depth, bits).unwrap());
        base.with_metadata(metadata, vec![integer_pixels(bits, depth)])
            .unwrap()
    };
    let a = make(rgba.clone(), depth.clone());
    rgba[0] ^= 1; // Invisible RGB change, including a real U16 low-bit change.
    let b = make(rgba.clone(), depth.clone());
    depth[14] ^= 1; // Extra-only change at the opposite canvas corner.
    let c = make(rgba, depth);
    vec![a.clone(), b, c.clone(), c, a]
}

fn combination_options(variant: usize) -> EncodeOptions {
    let mut result = options();
    if variant == 1 {
        result.threads = 4;
        result.modular.predictor = Predictor::Gradient;
        result.modular.fixed_tree = Some(FixedTree::GradientLeaf);
        result.modular.entropy = Coding::Ans;
        result.modular.rct = Some(6);
    }
    result
}

fn combination_matrix(
    mut visit: impl FnMut(&[Image], Timing, Reference, usize, &[PlannedFrame<'_>], &[u8]),
) {
    for bits in [8, 16] {
        for timing in [Timing::Layers, Timing::Mixed] {
            let images = combination_images(bits, timing);
            for reference in [
                Reference::Previous,
                Reference::First,
                Reference::SmallestCrop,
            ] {
                let plan = multiframe::plan(&images, &timing.durations(), reference).unwrap();
                for variant in 0..2 {
                    let frames: Vec<_> = plan.iter().map(PlannedFrame::as_frame).collect();
                    let bytes = rs_jxl::encode_frames(
                        &images[0],
                        &frames,
                        &combination_options(variant),
                    )
                    .unwrap_or_else(|error| {
                        panic!("bits={bits} {timing:?} {reference:?} variant={variant}: {error}")
                    });
                    visit(&images, timing, reference, variant, &plan, &bytes);
                }
            }
        }
    }
}

fn alpha_plane(image: &Image) -> PixelData {
    match image.pixels() {
        PixelData::U8(values) => PixelData::U8(values.chunks_exact(4).map(|p| p[3]).collect()),
        PixelData::U16(values) => PixelData::U16(values.chunks_exact(4).map(|p| p[3]).collect()),
        _ => panic!("integer fixture required"),
    }
}

fn assert_native_original(actual: &DecodedFrame, expected: &Image, duration: u32, label: &str) {
    assert_eq!(
        (actual.width, actual.height),
        (expected.width() as usize, expected.height() as usize),
        "{label}"
    );
    assert_eq!(actual.layout, expected.layout(), "{label}");
    assert_eq!(
        &actual.pixels,
        expected.pixels(),
        "{label}: original RGBA, including invisible RGB"
    );
    assert_eq!(actual.extra_planes.len(), 2, "{label}");
    assert_eq!(
        actual.extra_planes[0],
        alpha_plane(expected),
        "{label}: separate alpha"
    );
    assert_eq!(
        &actual.extra_planes[1..],
        expected.extra_planes(),
        "{label}: original depth"
    );
    assert_eq!(actual.duration_ticks, duration, "{label}");
    assert!(actual.complete, "{label}: complete frame");
}

fn assert_selected(actual: &DecodedFrame, expected: &DecodedFrame, label: &str) {
    assert_eq!(actual.index, expected.index, "{label}: selected index");
    assert_eq!(
        (actual.width, actual.height, actual.layout),
        (expected.width, expected.height, expected.layout),
        "{label}"
    );
    assert_eq!(actual.pixels, expected.pixels, "{label}: selected pixels");
    assert_eq!(
        actual.extra_planes, expected.extra_planes,
        "{label}: selected extras"
    );
    assert_eq!(actual.duration_ticks, expected.duration_ticks, "{label}");
    assert_eq!(actual.name, expected.name, "{label}");
    assert_eq!(
        actual.header.layer_info.crop_origin, expected.header.layer_info.crop_origin,
        "{label}"
    );
    assert_eq!(actual.header.is_last, expected.header.is_last, "{label}");
    assert_eq!(actual.complete, expected.complete, "{label}");
}

#[test]
fn integer_rgba_depth_native_combination_matrix() {
    combination_matrix(|images, timing, reference, variant, plan, bytes| {
        let durations = timing.durations();
        for threads in [1, 4] {
            for coalescing in [true, false] {
                let label = format!(
                    "bits={} {timing:?} {reference:?} variant={variant} threads={threads} coalescing={coalescing}",
                    images[0].metadata().bit_depth.bits_per_sample
                );
                let mut decoder = Decoder::new(DecodeOptions {
                    threads,
                    coalescing,
                    include_extra_channels: true,
                    ..Default::default()
                })
                .unwrap();
                let all = decoder
                    .decode(bytes)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                assert!(all.complete, "{label}: full stream");
                // Display indices enumerate visible states, not encoded layers.
                let indices = if coalescing {
                    timing.visible()
                } else {
                    (0..plan.len()).collect()
                };
                assert_eq!(all.frames.len(), indices.len(), "{label}");
                for (index, (&source, actual)) in indices.iter().zip(&all.frames).enumerate() {
                    let expected = if coalescing {
                        &images[source]
                    } else {
                        plan[source].image.as_ref()
                    };
                    assert_eq!(actual.index, index, "{label}");
                    assert_native_original(actual, expected, durations[source], &label);
                    if !coalescing {
                        let (x, y) = plan[source].options.origin;
                        assert_eq!(
                            actual.header.layer_info.crop_origin,
                            (i64::from(x), i64::from(y)),
                            "{label}: raw origin"
                        );
                    }
                    let selected = decoder
                        .decode_frame(bytes, index)
                        .unwrap_or_else(|error| panic!("{label}: selected {index}: {error}"));
                    assert!(
                        !selected.complete,
                        "{label}: selection does not validate tail"
                    );
                    assert_eq!(selected.frames.len(), 1, "{label}");
                    assert_selected(&selected.frames[0], actual, &label);
                }
            }
        }
    });
}

fn assert_original_depth(decoded: &Npy, images: &[Image], visible: &[usize], label: &str) {
    assert_original_colors(decoded, images, visible, label);
    let [_, height, width, channels] = decoded.shape;
    assert_eq!(channels, 5, "{label}: RGB, alpha, depth");
    assert_eq!(
        decoded.words.len(),
        visible.len() * height * width * channels,
        "{label}"
    );
    for (frame, &source) in visible.iter().enumerate() {
        let image = &images[source];
        assert_eq!(image.metadata().extra_channels[1].dim_shift, 0);
        let maximum = (1u32 << image.metadata().extra_channels[1].bit_depth.bits_per_sample) - 1;
        for pixel in 0..height * width {
            let expected = match &image.extra_planes()[0] {
                PixelData::U8(values) => u32::from(values[pixel]),
                PixelData::U16(values) => u32::from(values[pixel]),
                _ => panic!("integer depth required"),
            };
            let value =
                f32::from_bits(decoded.words[(frame * height * width + pixel) * channels + 4]);
            assert!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "{label}: normalized depth"
            );
            assert_eq!(
                (f64::from(value) * f64::from(maximum)).round(),
                f64::from(expected),
                "{label}: original depth frame {source}, pixel {pixel}"
            );
        }
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; integer RGBA/depth combination matrix"]
fn integer_rgba_depth_combinations_match_official_originals() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    combination_matrix(|images, timing, reference, variant, _, bytes| {
        let input = scratch.0.join("combinations.jxl");
        fs::write(&input, bytes).unwrap();
        for decoder in &decoders {
            let label = format!(
                "official bits={} {timing:?} {reference:?} variant={variant} {}",
                images[0].metadata().bit_depth.bits_per_sample,
                decoder.display()
            );
            let actual = decode(decoder, &input, &scratch, &label);
            assert_original_depth(&actual, images, &timing.visible(), &label);
        }
    });
}

fn original_npy(images: &[Image], visible: &[usize]) -> Npy {
    let mut words = Vec::new();
    for &source in visible {
        let image = &images[source];
        let values = |data: &PixelData| -> Vec<u16> {
            match data {
                PixelData::U8(v) => v.iter().copied().map(u16::from).collect(),
                PixelData::U16(v) => v.clone(),
                _ => panic!("integer fixture required"),
            }
        };
        let rgba = values(image.pixels());
        let depth = values(&image.extra_planes()[0]);
        let maximum = ((1u32 << image.metadata().bit_depth.bits_per_sample) - 1) as f32;
        for (pixel, &extra) in rgba.chunks_exact(4).zip(&depth) {
            words.extend(
                pixel
                    .iter()
                    .chain(std::iter::once(&extra))
                    .map(|&v| (f32::from(v) / maximum).to_bits()),
            );
        }
    }
    Npy {
        shape: [visible.len(), 3, 5, 5],
        words,
    }
}

#[test]
fn combination_fixture_and_integer_oracle_self_checks() {
    assert_eq!(Timing::Layers.visible(), [4]);
    assert_eq!(Timing::Mixed.visible(), [1, 3, 4]);
    for bits in [8, 16] {
        let images = combination_images(bits, Timing::Mixed);
        assert_eq!(images[0].pixels(), images[4].pixels());
        assert_eq!(images[2].pixels(), images[3].pixels());
        assert_eq!(images[2].extra_planes(), images[3].extra_planes());
        assert_eq!(images[1].pixels(), images[2].pixels());
        assert_ne!(images[1].extra_planes(), images[2].extra_planes());
        assert_eq!(
            multiframe::difference_bounds(&images[0], &images[1]).unwrap(),
            Some(Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1
            })
        );
        assert_eq!(
            multiframe::difference_bounds(&images[1], &images[2]).unwrap(),
            Some(Rect {
                x: 4,
                y: 2,
                width: 1,
                height: 1
            })
        );
        let plan =
            multiframe::plan(&images, &Timing::Mixed.durations(), Reference::SmallestCrop).unwrap();
        assert_eq!(
            plan[2].options.blend.source, 2,
            "previous has smaller depth-only crop"
        );
        assert_eq!(
            plan[4].options.blend.source, 1,
            "return to first has smallest crop"
        );
        assert!(plan[3].changed.is_none(), "repeated state");
        if let PixelData::U16(values) = images[0].pixels() {
            assert!(
                values.iter().any(|v| v % 257 != 0),
                "U16 fixture must exercise low bits"
            );
            assert!(
                values
                    .chunks_exact(4)
                    .any(|p| p[3] != 0 && p[3] != u16::MAX)
            );
        }
        let visible = Timing::Mixed.visible();
        let expected = original_npy(&images, &visible);
        assert_original_depth(&expected, &images, &visible, "synthetic original");
        // Test the oracle itself against corrupt color, alpha, extra and frame ordering.
        for sample in [0, 3, 4, 5 * 15 + 4] {
            let mut corrupt = original_npy(&images, &visible);
            corrupt.words[sample] = f32::NAN.to_bits();
            assert!(
                std::panic::catch_unwind(|| assert_original_depth(
                    &corrupt,
                    &images,
                    &visible,
                    "deliberate corruption"
                ))
                .is_err()
            );
        }
        let swapped = original_npy(&images, &[3, 1, 4]);
        assert!(
            std::panic::catch_unwind(|| assert_original_depth(
                &swapped,
                &images,
                &visible,
                "swapped states"
            ))
            .is_err()
        );
        let mut wrong_depth = original_npy(&images, &visible);
        wrong_depth.words[4] = (4.0f32 / ((1u32 << bits) - 1) as f32).to_bits();
        assert!(
            std::panic::catch_unwind(|| assert_original_depth(
                &wrong_depth,
                &images,
                &visible,
                "one depth integer changed"
            ))
            .is_err()
        );
    }
}
