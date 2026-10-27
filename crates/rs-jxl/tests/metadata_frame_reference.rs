//! Metadata/frame regressions with independent official full-frame oracles.
//!
//! Set RS_JXL_DJXL to an OS path list of official djxl executables, then run:
//! cargo test -p rs-jxl --test metadata_frame_reference -- --ignored --nocapture
//! Non-ignored tests need no external decoder. No other test file is included.
//!
//! The reference sequence is constructed directly, never through plan/crop.
//! Official NPY output contains every coalesced frame and every extra channel
//! (lib/extras/enc/npy.cc at b866b3d). Compare decoded f32 words, not JXL bytes.
//! Integer color samples are also recovered exactly at 8/16 bits. Upsampled
//! extras are compared to the full-frame decode, not to an invented filter.

use std::env;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::entropy::stream::Coding;
use rs_jxl::frame::{BlendInfo, BlendMode, Frame, FrameOptions};
use rs_jxl::metadata::{Animation, BitDepth, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::multiframe::{self, PlannedFrame, Rect, Reference};
use rs_jxl::{EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout};

const FRAME_COUNT: usize = 5;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
const REFERENCES: [Reference; 4] = [
    Reference::Full,
    Reference::Previous,
    Reference::First,
    Reference::SmallestCrop,
];
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug)]
enum Samples {
    U8,
    U16,
    F32,
    CustomFloat,
}

impl Samples {
    fn depth(self) -> BitDepth {
        match self {
            Self::U8 => BitDepth::new(8).unwrap(),
            Self::U16 => BitDepth::new(16).unwrap(),
            Self::F32 => BitDepth::floating(32, 8).unwrap(),
            Self::CustomFloat => BitDepth::floating(16, 6).unwrap(),
        }
    }

    fn pixels(self, values: Vec<u8>) -> PixelData {
        match self {
            Self::U8 => PixelData::U8(values),
            Self::U16 => PixelData::U16(values.into_iter().map(|v| u16::from(v) * 257).collect()),
            // Multiples of 1/32 are exactly representable in both float layouts.
            Self::F32 | Self::CustomFloat => PixelData::F32(
                values
                    .into_iter()
                    .map(|v| f32::from(v % 33) / 32.0)
                    .collect(),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Timing {
    Layers,
    AnimatedLayers,
    Animation,
    Mixed,
}

impl Timing {
    fn durations(self) -> [u32; FRAME_COUNT] {
        match self {
            Self::Layers | Self::AnimatedLayers => [0; FRAME_COUNT],
            Self::Animation => [2, 3, 5, 7, 11],
            Self::Mixed => [0, 2, 0, 3, 0],
        }
    }

    fn visible(self) -> Vec<usize> {
        self.durations()
            .into_iter()
            .enumerate()
            .filter_map(|(i, duration)| (duration != 0 || i + 1 == FRAME_COUNT).then_some(i))
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    width: u32,
    height: u32,
    shift: u8,
    layout: PixelLayout,
    samples: Samples,
    timing: Timing,
}

fn cases(samples: &[Samples], mut visit: impl FnMut(Case)) {
    for &samples in samples {
        for layout in [PixelLayout::Gray, PixelLayout::Rgba] {
            // 7x5/shift=3 has exactly one extra sample. Keep that edge case,
            // and add 17x9 so the 8x upsampler also sees a nonconstant plane.
            for (width, height, shift) in [(7, 5, 1), (7, 5, 2), (7, 5, 3), (17, 9, 3)] {
                for timing in [
                    Timing::Layers,
                    Timing::AnimatedLayers,
                    Timing::Animation,
                    Timing::Mixed,
                ] {
                    visit(Case {
                        width,
                        height,
                        shift,
                        layout,
                        samples,
                        timing,
                    });
                }
            }
        }
    }
}

fn fixture(case: Case) -> Vec<Image> {
    let count = case.width as usize * case.height as usize;
    let channels = case.layout.channels();
    let mut color: Vec<u8> = (0..count * channels)
        .map(|i| ((i * 37 + i / channels * 11 + 19) % 251) as u8)
        .collect();
    if case.layout.has_alpha() {
        for pixel in 0..count {
            color[pixel * channels + channels - 1] = if pixel % 3 == 0 { 0 } else { 255 };
        }
    }
    let scale = 1usize << case.shift;
    let ew = (case.width as usize).div_ceil(scale);
    let eh = (case.height as usize).div_ceil(scale);
    let mut extra: Vec<u8> = (0..ew * eh)
        .map(|i| [16, 32, 0, 8][(i % ew + 2 * (i / ew)) % 4])
        .collect();
    assert!(extra.len() == 1 || extra.iter().any(|&v| v != extra[0]));

    let make = |color: Vec<u8>, extra: Vec<u8>| {
        let base = Image::new(
            case.width,
            case.height,
            case.layout,
            case.samples.pixels(color),
        )
        .unwrap();
        let mut metadata = base.metadata().clone();
        metadata.bit_depth = case.samples.depth();
        if case.layout.has_alpha() {
            metadata.extra_channels[0].bit_depth = case.samples.depth();
        }
        if !matches!(case.timing, Timing::Layers) {
            metadata.animation = Some(Animation::default());
        }
        let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
        info.bit_depth = case.samples.depth();
        info.dim_shift = case.shift;
        metadata.extra_channels.push(info);
        base.with_metadata(metadata, vec![case.samples.pixels(extra)])
            .unwrap()
    };
    let a = make(color.clone(), extra.clone());
    // Change a color under zero alpha, without changing any extra samples.
    color[0] ^= 1;
    let b = make(color.clone(), extra.clone());
    // Change only the extra, including the incomplete bottom-right cell.
    extra[0] ^= 8;
    if extra.len() > 1 {
        *extra.last_mut().unwrap() ^= 16;
    }
    let c = make(color, extra);
    vec![a.clone(), a, b, c.clone(), c]
}

fn options() -> EncodeOptions {
    EncodeOptions {
        max_output_bytes: 4 << 20,
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

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; byte-selected automatic crop/mask/multiple layers"]
fn automatic_byte_selected_layers_match_official_decoders() {
    use rs_jxl::multiframe::{AutoOptions, Strategy, encode_auto};
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let mut mask_winners = 0;
    let mut tile_winners = 0;
    let mut checks = 0;
    for (bits, alpha_bits) in [(8, 8), (16, 16), (8, 16), (16, 8), (12, 1)] {
        for transparent in [false, true] {
            let (w, h) = (49usize, 37usize);
            let maximum = (1u32 << bits) - 1;
            let alpha_maximum = (1u32 << alpha_bits) - 1;
            let mut base: Vec<u16> = (0..w * h * 4)
                .map(|i| ((i * 1237 + i / 17 * 2017) as u32 & maximum) as u16)
                .collect();
            for (i, p) in base.chunks_exact_mut(4).enumerate() {
                p[3] = if transparent && i % 2 == 0 {
                    0
                } else {
                    alpha_maximum as u16
                };
            }
            let images: Vec<_> = (0..5)
                .map(|index| {
                    let mut values = base.clone();
                    if index == 1 || index == 2 {
                        values[(2 * w + 2) * 4] ^= 0x55;
                        values[((h - 3) * w + w - 3) * 4] ^= 0x55;
                    }
                    let pixels = if bits <= 8 && alpha_bits <= 8 {
                        PixelData::U8(values.into_iter().map(|v| v as u8).collect())
                    } else {
                        PixelData::U16(values)
                    };
                    let image = Image::new(w as u32, h as u32, PixelLayout::Rgba, pixels).unwrap();
                    let mut metadata = image.metadata().clone();
                    metadata.bit_depth = BitDepth::new(bits).unwrap();
                    metadata.extra_channels[0].bit_depth = BitDepth::new(alpha_bits).unwrap();
                    metadata.animation = Some(Animation::default());
                    image.with_metadata(metadata, vec![]).unwrap()
                })
                .collect();
            for timing in [Timing::Animation, Timing::Mixed] {
                let durations = timing.durations();
                let frames: Vec<_> = images
                    .iter()
                    .zip(durations)
                    .map(|(image, duration)| Frame {
                        image,
                        options: FrameOptions {
                            duration,
                            ..Default::default()
                        },
                    })
                    .collect();
                let full = rs_jxl::encode_frames(&images[0], &frames, &options()).unwrap();
                fs::write(scratch.0.join("auto-full.jxl"), full).unwrap();
                for reference in [
                    Reference::Previous,
                    Reference::First,
                    Reference::SmallestCrop,
                ] {
                    for allow_multi_rect in [false, true] {
                        let result = encode_auto(
                            &images,
                            &durations,
                            &options(),
                            &AutoOptions {
                                reference,
                                allow_multi_rect,
                                tile_size: 8,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                        for frame in &result.frames {
                            match frame.candidates[frame.winner_index].strategy {
                                Strategy::MaskPrevious | Strategy::MaskFirst => mask_winners += 1,
                                Strategy::TilesPrevious | Strategy::TilesFirst => tile_winners += 1,
                                _ => {}
                            }
                        }
                        let input = scratch.0.join("auto.jxl");
                        fs::write(&input, result.bytes).unwrap();
                        for decoder in &decoders {
                            let context = format!(
                                "auto color={bits} alpha={alpha_bits} transparent={transparent} {timing:?} {reference:?} multi={allow_multi_rect}"
                            );
                            let expected = decode(
                                decoder,
                                &scratch.0.join("auto-full.jxl"),
                                &scratch,
                                &context,
                            );
                            let actual = decode(decoder, &input, &scratch, &context);
                            assert_decoded_equal(&actual, &expected, &context);
                            assert_original_colors(&actual, &images, &timing.visible(), &context);
                            checks += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        mask_winners > 0,
        "mask route must actually win and be decoded"
    );
    assert!(
        tile_winners > 0,
        "multiple-layer route must actually win and be decoded"
    );
    eprintln!(
        "automatic strategies: {checks} official exact-sample comparisons; {mask_winners} mask and {tile_winners} tile winners"
    );
}

#[test]
#[ignore = "requires RS_JXL_DJXL; new merged rectangles must actually win"]
fn merged_rectangles_match_official_decoders() {
    use rs_jxl::multiframe::{AutoOptions, Strategy, encode_auto};
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for bits in [8, 16] {
        let (width, height) = (129usize, 97usize);
        let mut state = 0x1234_5678u32;
        let base: Vec<u16> = (0..width * height)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state & ((1 << bits) - 1)) as u16
            })
            .collect();
        let mut changed = base.clone();
        for (left, top) in [(3, 3), (110, 78)] {
            for y in 0..3 {
                for x in 0..3 {
                    changed[(top + y * 4) * width + left + x * 4] ^= 0x55;
                }
            }
        }
        let images: Vec<_> = [base, changed]
            .into_iter()
            .map(|values| {
                let pixels = if bits == 8 {
                    PixelData::U8(values.into_iter().map(|v| v as u8).collect())
                } else {
                    PixelData::U16(values)
                };
                let image =
                    Image::new(width as u32, height as u32, PixelLayout::Gray, pixels).unwrap();
                let mut metadata = image.metadata().clone();
                metadata.animation = Some(Animation::default());
                image.with_metadata(metadata, vec![]).unwrap()
            })
            .collect();
        for durations in [[1, 2], [0, 0]] {
            let frames: Vec<_> = images
                .iter()
                .zip(durations)
                .map(|(image, duration)| Frame {
                    image,
                    options: FrameOptions {
                        duration,
                        ..Default::default()
                    },
                })
                .collect();
            let full_path = scratch.0.join("merged-full.jxl");
            fs::write(
                &full_path,
                rs_jxl::encode_frames(&images[0], &frames, &options()).unwrap(),
            )
            .unwrap();
            for reference in [Reference::Previous, Reference::First] {
                let result = encode_auto(
                    &images,
                    &durations,
                    &options(),
                    &AutoOptions {
                        reference,
                        tile_size: 4,
                        max_rectangles: 2,
                        ..Default::default()
                    },
                )
                .unwrap();
                let report = &result.frames[1];
                let winner = &report.candidates[report.winner_index];
                assert!(matches!(
                    winner.strategy,
                    Strategy::MergedPrevious | Strategy::MergedFirst
                ));
                assert_eq!(winner.encoded_frames, 2);
                let path = scratch.0.join("merged.jxl");
                fs::write(&path, result.bytes).unwrap();
                let visible = if durations[0] == 0 {
                    vec![1]
                } else {
                    vec![0, 1]
                };
                for decoder in &decoders {
                    let context = format!("merged {bits} {durations:?} {reference:?}");
                    let expected = decode(decoder, &full_path, &scratch, &context);
                    let actual = decode(decoder, &path, &scratch, &context);
                    assert_decoded_equal(&actual, &expected, &context);
                    assert_original_colors(&actual, &images, &visible, &context);
                }
            }
        }
    }
}

fn assert_full_plan(images: &[Image], durations: &[u32], plan: &[PlannedFrame<'_>]) {
    assert_eq!(plan.len(), images.len());
    for (i, planned) in plan.iter().enumerate() {
        assert_eq!(
            (planned.image.width(), planned.image.height()),
            (images[i].width(), images[i].height()),
            "frame {i}: subsampled fallback must not crop, including identical frames"
        );
        assert_eq!(
            planned.image.pixels(),
            images[i].pixels(),
            "frame {i}: color/alpha storage"
        );
        assert_eq!(
            planned.image.extra_planes(),
            images[i].extra_planes(),
            "frame {i}: extra storage"
        );
        assert_eq!(
            planned.image.metadata(),
            images[i].metadata(),
            "frame {i}: metadata"
        );
        assert_eq!(planned.options.origin, (0, 0), "frame {i}");
        assert_eq!(
            planned.options.duration, durations[i],
            "frame {i}: do not replace zero durations"
        );
        assert_eq!(planned.options.blend.mode, BlendMode::Replace, "frame {i}");
        assert!(
            !planned.options.reference_only,
            "frame {i}: visible/layer semantics"
        );
    }
}

#[test]
fn subsampled_plans_keep_full_images_including_identical_frames() {
    cases(
        &[
            Samples::U8,
            Samples::U16,
            Samples::F32,
            Samples::CustomFloat,
        ],
        |case| {
            let images = fixture(case);
            let durations = case.timing.durations();
            assert_eq!(
                multiframe::difference_bounds(&images[0], &images[1]).unwrap(),
                None
            );
            assert_eq!(
                multiframe::difference_bounds(&images[3], &images[4]).unwrap(),
                None
            );
            assert_ne!(images[1].pixels(), images[2].pixels());
            assert_eq!(images[1].extra_planes(), images[2].extra_planes());
            assert_eq!(images[2].pixels(), images[3].pixels());
            assert_ne!(images[2].extra_planes(), images[3].extra_planes());
            for reference in REFERENCES {
                let plan = multiframe::plan(&images, &durations, reference)
                    .unwrap_or_else(|error| panic!("{case:?}/{reference:?}: {error}"));
                assert_full_plan(&images, &durations, &plan);
            }
        },
    );
}

#[derive(Debug, PartialEq, Eq)]
struct Npy {
    shape: [usize; 4],
    words: Vec<u32>,
}

// Only the fixed ASCII dictionary emitted by official enc/npy.cc, not a
// general Python-literal parser. No dependency on float_reference.rs helpers.
fn parse_npy(bytes: &[u8]) -> Result<Npy, String> {
    if bytes.len() < 10 || &bytes[..8] != b"\x93NUMPY\x01\x00" {
        return Err("expected NPY version 1.0".into());
    }
    let end = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let header = bytes.get(10..end).ok_or("truncated NPY header")?;
    if !header.is_ascii() || !header.ends_with(b"\n") {
        return Err("expected newline-terminated ASCII NPY header".into());
    }
    let header = std::str::from_utf8(header)
        .map_err(|_| "invalid NPY header")?
        .trim_end();
    let dimensions = header
        .strip_prefix("{'descr': '<f4', 'fortran_order': False, 'shape': (")
        .and_then(|s| s.strip_suffix("), }"))
        .ok_or("expected official C-order <f4 NPY dictionary")?;
    let dimensions: Vec<_> = dimensions.split(',').map(str::trim).collect();
    if dimensions.len() != 4 {
        return Err("expected four NPY dimensions".into());
    }
    let mut shape = [0usize; 4];
    let mut elements = 1usize;
    for (out, dimension) in shape.iter_mut().zip(dimensions) {
        if dimension.is_empty() || !dimension.bytes().all(|b| b.is_ascii_digit()) {
            return Err("invalid NPY dimension".into());
        }
        *out = dimension.parse().map_err(|_| "NPY dimension overflow")?;
        if *out == 0 {
            return Err("empty NPY dimension".into());
        }
        elements = elements
            .checked_mul(*out)
            .ok_or("NPY element count overflow")?;
    }
    let payload = &bytes[end..];
    if elements.checked_mul(4) != Some(payload.len()) {
        return Err("NPY payload length does not match shape".into());
    }
    Ok(Npy {
        shape,
        words: payload
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect(),
    })
}

#[test]
fn npy_oracle_checks_shape_payload_and_float_words() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 1, 1, 2), }\n";
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    let words: [u32; 4] = [0, 0x8000_0000, 0x7fc1_2345, 0x3f80_0000];
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    assert_eq!(
        parse_npy(&bytes).unwrap(),
        Npy {
            shape: [2, 1, 1, 2],
            words: words.to_vec()
        }
    );
    for end in 0..bytes.len() {
        assert!(parse_npy(&bytes[..end]).is_err(), "truncated at {end}");
    }
    let mut surplus = bytes.clone();
    surplus.push(0);
    assert!(parse_npy(&surplus).is_err());
    let mut empty_shape = bytes;
    let shape = 10 + header.find("(2,").unwrap() + 1;
    empty_shape[shape] = b'0';
    assert!(parse_npy(&empty_shape).is_err());
}

fn assert_original_colors(decoded: &Npy, images: &[Image], visible: &[usize], context: &str) {
    let image = &images[0];
    let pixels = image.width() as usize * image.height() as usize;
    let channels = image.channel_count();
    assert_eq!(
        decoded.shape,
        [
            visible.len(),
            image.height() as usize,
            image.width() as usize,
            channels
        ],
        "{context}: coalesced frames/channels"
    );
    for (frame, &index) in visible.iter().enumerate() {
        let image = &images[index];
        for pixel in 0..pixels {
            for c in 0..image.layout().channels() {
                let word = decoded.words[(frame * pixels + pixel) * channels + c];
                let source = pixel * image.layout().channels() + c;
                let integer = match image.pixels() {
                    PixelData::U8(values) => Some(u32::from(values[source])),
                    PixelData::U16(values) => Some(u32::from(values[source])),
                    PixelData::F32(values) => {
                        assert_eq!(
                            word,
                            values[source].to_bits(),
                            "{context}: frame {index}, pixel {pixel}, channel {c}, original float words"
                        );
                        None
                    }
                    _ => panic!("unexpected fixture storage"),
                };
                if let Some(expected) = integer {
                    let maximum = (1u32 << image.channel_depth(c).bits_per_sample) - 1;
                    let value = f32::from_bits(word);
                    assert!(
                        value.is_finite() && (0.0..=1.0).contains(&value),
                        "{context}: normalized integer color"
                    );
                    assert_eq!(
                        (f64::from(value) * f64::from(maximum)).round(),
                        f64::from(expected),
                        "{context}: frame {index}, pixel {pixel}, channel {c}, exact integer recovery"
                    );
                }
            }
        }
    }
}

fn assert_decoded_equal(actual: &Npy, expected: &Npy, context: &str) {
    assert_eq!(actual.shape, expected.shape, "{context}: decoded shape");
    assert_eq!(
        actual.words.len(),
        expected.words.len(),
        "{context}: word count"
    );
    let [_, height, width, channels] = expected.shape;
    for (i, (&actual, &expected)) in actual.words.iter().zip(&expected.words).enumerate() {
        assert_eq!(
            actual,
            expected,
            "{context}: frame {}, x {}, y {}, channel {}; expected {expected:08x}, got {actual:08x}",
            i / (height * width * channels),
            i / channels % width,
            i / (width * channels) % height,
            i % channels
        );
    }
}

fn official_matrix(samples: &[Samples]) {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let mut checks = 0usize;
    cases(samples, |case| {
        let images = fixture(case);
        let durations = case.timing.durations();
        let visible = case.timing.visible();
        // No planner, crops, references, or copied planner options in the oracle.
        let full_frames: Vec<_> = images
            .iter()
            .zip(durations)
            .map(|(image, duration)| Frame {
                image,
                options: FrameOptions {
                    duration,
                    ..Default::default()
                },
            })
            .collect();
        let baseline = rs_jxl::encode_frames(&images[0], &full_frames, &options())
            .unwrap_or_else(|error| panic!("{case:?}: independent full-frame encode: {error}"));
        let full_path = scratch.0.join("full.jxl");
        fs::write(&full_path, baseline).unwrap();
        for decoder in &decoders {
            let label = format!("{case:?}/{}", decoder.display());
            let expected = decode(decoder, &full_path, &scratch, &label);
            assert_original_colors(&expected, &images, &visible, &label);
            for reference in REFERENCES {
                let label = format!("{label}/{reference:?}");
                let plan = multiframe::plan(&images, &durations, reference).unwrap();
                assert_full_plan(&images, &durations, &plan);
                let frames: Vec<_> = plan.iter().map(PlannedFrame::as_frame).collect();
                let bytes = rs_jxl::encode_frames(&images[0], &frames, &options())
                    .unwrap_or_else(|error| panic!("{label}: planned encode: {error}"));
                let input = scratch.0.join("planned.jxl");
                fs::write(&input, bytes).unwrap();
                let actual = decode(decoder, &input, &scratch, &label);
                assert_decoded_equal(&actual, &expected, &label);
                checks += 1;
            }
        }
    });
    eprintln!(
        "{checks} official metadata/frame comparisons passed (all coalesced color/extra words)"
    );
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn integer_subsampled_auto_plans_match_independent_full_frame_decodes() {
    official_matrix(&[Samples::U8, Samples::U16]);
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list; finite F32 and custom 16/6 metadata"]
fn float_subsampled_auto_plans_match_independent_full_frame_decodes() {
    official_matrix(&[Samples::F32, Samples::CustomFloat]);
}

#[test]
#[ignore = "requires RS_JXL_DJXL; patch atlas with native-size mixed extra planes"]
fn patch_modes_preserve_mixed_extra_depths_and_subsampled_words() {
    use rs_jxl::tools::patches::{self, ImageView, PatchOptions, SourcePatchConfig};
    let scratch = ScratchDir::new();
    let decoders = decoder_paths();
    let (w, h) = (129usize, 83usize);
    for bits in [8u8, 16] {
        let maximum = (1u32 << bits) - 1;
        let mut samples = Vec::new();
        let mut colors = vec![Vec::new(); 3];
        for y in 0..h {
            for x in 0..w {
                let glyph = [(17, 17), (81, 17), (33, 53), (97, 53)]
                    .iter()
                    .any(|&(px, py)| (px..px + 6).contains(&x) && (py..py + 5).contains(&y));
                let v = if glyph { maximum / 4 } else { maximum };
                for plane in &mut colors {
                    plane.push(v as i32);
                    samples.push(v as u16);
                }
                samples.push(0); // All RGB remains meaningful under zero alpha.
            }
        }
        let pixels = if bits == 8 {
            PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
        } else {
            PixelData::U16(samples)
        };
        let original = Image::new(w as u32, h as u32, PixelLayout::Rgba, pixels).unwrap();
        let mut metadata = original.metadata().clone();
        let mut extras = Vec::new();
        for (depth, shift) in [
            (BitDepth::new(12).unwrap(), 0),
            (BitDepth::new(16).unwrap(), 1),
            (BitDepth::floating(16, 5).unwrap(), 2),
            (BitDepth::floating(32, 8).unwrap(), 0),
            (BitDepth::new(8).unwrap(), 3),
        ] {
            let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
            info.bit_depth = depth;
            info.dim_shift = shift;
            metadata.extra_channels.push(info);
            let count = w.div_ceil(1 << shift) * h.div_ceil(1 << shift);
            extras.push(match (depth.bits_per_sample, depth.is_float()) {
                (16, true) => PixelData::F16(
                    (0..count)
                        .map(|i| [0, 0x3400, 0x3800, 0x3c00][i % 4])
                        .collect(),
                ),
                (32, true) => {
                    PixelData::F32((0..count).map(|i| [0.0, -0.0, 0.5, 1.0][i % 4]).collect())
                }
                (8, false) => PixelData::U8((0..count).map(|i| (i % 253) as u8).collect()),
                (_, false) => {
                    PixelData::U16((0..count).map(|i| ((i * 13) % 4096) as u16).collect())
                }
                _ => unreachable!(),
            });
        }
        let original = original.with_metadata(metadata, extras).unwrap();
        let config = SourcePatchConfig::default();
        assert!(
            patches::discover_source(
                ImageView {
                    width: w,
                    height: h,
                    bit_depth: u32::from(bits),
                    color_channels: 3,
                    planes: &colors
                },
                &config
            )
            .unwrap()
            .is_some(),
            "fixture must exercise source Add"
        );
        let baseline = EncodeOptions {
            threads: 2,
            ..EncodeOptions::default()
        };
        let full_path = scratch.0.join("patch-baseline.jxl");
        fs::write(&full_path, rs_jxl::encode(&original, &baseline).unwrap()).unwrap();
        for mode in [0, 1] {
            let mut options = baseline.clone();
            if mode == 0 {
                options.source_patches = Some(config.clone());
            } else {
                options.patches = Some(PatchOptions::default());
            }
            let path = scratch.0.join("patch-mixed.jxl");
            fs::write(&path, rs_jxl::encode(&original, &options).unwrap()).unwrap();
            for decoder in &decoders {
                let label = format!(
                    "mixed patch mode={mode} bits={bits} decoder={}",
                    decoder.display()
                );
                let expected = decode(decoder, &full_path, &scratch, &label);
                let actual = decode(decoder, &path, &scratch, &label);
                assert_original_colors(&actual, std::slice::from_ref(&original), &[0], &label);
                assert_original_colors(&expected, std::slice::from_ref(&original), &[0], &label);
                assert_eq!(actual.shape, expected.shape, "{label}");
                // Add rounds normalized integer color in decoder f32 arithmetic.
                // Above we recover and compare the original integer samples.
                // Extras are None-blended and retain their decoded float words,
                // including subsampled interpolation and float signed zero.
                for (i, (&a, &b)) in actual.words.iter().zip(&expected.words).enumerate() {
                    if i % original.channel_count() >= original.color_channels() {
                        assert_eq!(a, b, "{label}: extra output sample {i}");
                    }
                }
            }
        }
    }
}

fn reference_images(animated: bool) -> (Image, Image) {
    let images = fixture(Case {
        width: 7,
        height: 5,
        shift: 0,
        layout: PixelLayout::Rgba,
        samples: Samples::U8,
        timing: if animated {
            Timing::Animation
        } else {
            Timing::Layers
        },
    });
    let full = images[0].clone();
    let crop = multiframe::crop(
        &images[2],
        Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
    )
    .unwrap();
    (full, crop)
}

fn assert_invalid_reference(canvas: &Image, frames: &[Frame<'_>]) {
    let mut sink = vec![19, 23, 29];
    let error = rs_jxl::encode_frames_to(canvas, frames, &options(), &mut sink).unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput(_)),
        "expected input validation, got {error}"
    );
    assert_eq!(
        sink,
        [19, 23, 29],
        "validation must not publish a partial codestream"
    );
}

#[test]
fn pre_transform_references_cannot_feed_color_or_extra_blending() {
    let (full, crop) = reference_images(false);
    for slot in 0..=3 {
        for reference_only in [false, true] {
            for kind in 0..3 {
                let mut blend = FrameOptions::default();
                let image = if kind == 0 { &crop } else { &full };
                match kind {
                    0 => {
                        blend.origin = (2, 1);
                        blend.blend.source = slot;
                    }
                    1 => {
                        blend.blend = BlendInfo {
                            mode: BlendMode::Add,
                            source: slot,
                            ..Default::default()
                        }
                    }
                    // Color and alpha Replace need no reference; only depth Add does.
                    _ => {
                        blend.extra_blend =
                            vec![BlendInfo::default(); full.metadata().extra_channels.len()];
                        *blend.extra_blend.last_mut().unwrap() = BlendInfo {
                            mode: BlendMode::Add,
                            source: slot,
                            ..Default::default()
                        };
                    }
                }
                let mut frames = [
                    Frame {
                        image: &full,
                        options: FrameOptions {
                            save_as_reference: slot,
                            reference_only,
                            save_before_color_transform: true,
                            ..Default::default()
                        },
                    },
                    Frame {
                        image,
                        options: blend,
                    },
                ];
                assert_invalid_reference(&full, &frames);
                frames[0].options.save_before_color_transform = false;
                assert!(
                    rs_jxl::encode_frames(&full, &frames, &options()).is_ok(),
                    "post-transform control: slot={slot}, reference_only={reference_only}, kind={kind}"
                );
            }
        }
    }
}

#[test]
fn overwritten_reference_stage_is_not_stale() {
    let (full, crop) = reference_images(false);
    for before in [false, true] {
        let frames = [
            Frame {
                image: &full,
                options: FrameOptions {
                    save_as_reference: 1,
                    save_before_color_transform: !before,
                    ..Default::default()
                },
            },
            Frame {
                image: &full,
                options: FrameOptions {
                    save_as_reference: 1,
                    save_before_color_transform: before,
                    ..Default::default()
                },
            },
            Frame {
                image: &crop,
                options: FrameOptions {
                    blend: BlendInfo {
                        source: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
        ];
        if before {
            assert_invalid_reference(&full, &frames);
        } else {
            assert!(rs_jxl::encode_frames(&full, &frames, &options()).is_ok());
        }
    }
}

#[test]
fn zero_duration_slot_zero_is_saved_but_timed_slot_zero_is_not() {
    let (full, crop) = reference_images(true);
    for duration in [0, 1] {
        let frames = [
            Frame {
                image: &full,
                options: FrameOptions {
                    duration,
                    ..Default::default()
                },
            },
            Frame {
                image: &crop,
                options: FrameOptions {
                    duration: 1,
                    ..Default::default()
                },
            },
        ];
        if duration == 0 {
            assert!(rs_jxl::encode_frames(&full, &frames, &options()).is_ok());
        } else {
            assert_invalid_reference(&full, &frames);
        }
    }
}

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Self {
        let root = env::temp_dir()
            .canonicalize()
            .expect("resolve temporary directory");
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        for _ in 0..100 {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "rs-jxl-metadata-frame-{}-{timestamp}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        }
        panic!("could not create a unique metadata-frame test directory");
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            if thread::panicking() {
                eprintln!("cleanup {}: {error}", self.0.display());
            } else {
                panic!("cleanup {}: {error}", self.0.display());
            }
        }
    }
}

struct ReapOnDrop(Child);

impl Drop for ReapOnDrop {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

fn wait_with_timeout(command: &mut Command) -> io::Result<Option<ExitStatus>> {
    let mut child = ReapOnDrop(command.spawn()?);
    let start = Instant::now();
    loop {
        if let Some(status) = child.0.try_wait()? {
            return Ok(Some(status));
        }
        if start.elapsed() >= COMMAND_TIMEOUT {
            child.0.kill()?;
            child.0.wait()?;
            return Ok(None);
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn decoder_paths() -> Vec<PathBuf> {
    let value = env::var_os("RS_JXL_DJXL").expect(
        "explicit oracle test requires RS_JXL_DJXL: an OS path list of official djxl executables",
    );
    let paths: Vec<_> = env::split_paths(&value).collect();
    assert!(!paths.is_empty(), "RS_JXL_DJXL must not be empty");
    paths
        .into_iter()
        .map(|path| {
            assert!(
                !path.as_os_str().is_empty() && path.is_file(),
                "invalid djxl executable: {}",
                path.display()
            );
            path.canonicalize()
                .unwrap_or_else(|error| panic!("resolve {}: {error}", path.display()))
        })
        .collect()
}

fn decode(decoder: &Path, input: &Path, scratch: &ScratchDir, context: &str) -> Npy {
    let output = scratch.0.join("output.npy");
    assert!(!output.exists(), "stale decoder output");
    let stdout = scratch.0.join("stdout.log");
    let stderr = scratch.0.join("stderr.log");
    // NPY interleaves extras and packs all frames in one file; unlike PNG it
    // needs neither --output_extra_channels nor filename enumeration. Leave
    // coalescing on and do not request color conversion or alpha compositing.
    let mut command = Command::new(decoder);
    command
        .arg(input)
        .arg(&output)
        .arg("--output_frames")
        .arg("--num_threads=1")
        .arg("--quiet")
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    let status = wait_with_timeout(&mut command);
    // Release redirection handles before cleanup on Windows.
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let diagnostics = format!(
        "stdout:\n{}\nstderr:\n{}",
        fs::read_to_string(stdout).unwrap_or_default(),
        fs::read_to_string(stderr).unwrap_or_default()
    );
    match status {
        Ok(Some(status)) if status.success() => (),
        Ok(Some(status)) => panic!("{context}: djxl exited {status}\n{diagnostics}"),
        Ok(None) => {
            panic!("{context}: djxl exceeded {COMMAND_TIMEOUT:?}; killed and reaped\n{diagnostics}")
        }
        Err(error) => panic!(
            "{context}: launch/poll {}: {error}\n{diagnostics}",
            decoder.display()
        ),
    }
    let bytes = fs::read(&output).unwrap_or_else(|error| panic!("{context}: read NPY: {error}"));
    let result = parse_npy(&bytes).unwrap_or_else(|error| panic!("{context}: {error}"));
    fs::remove_file(output).unwrap();
    result
}
