//! Focused public-API acceptance gaps, not a second tool/oracle matrix.
//! Parent runs this target centrally in debug and release. See
//! docs/PIXEL_ACCEPTANCE_AUDIT.md for regression provenance and oracle limits.

use std::io::{self, Write};

use rs_jxl::entropy::stream::Coding;
use rs_jxl::frame::{BlendInfo, BlendMode, Frame, FrameOptions};
use rs_jxl::metadata::{Animation, BitDepth, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::multiframe::{self, Rect};
use rs_jxl::tools::patches::{PatchOptions, SourcePatchConfig};
use rs_jxl::{EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout};

fn options() -> EncodeOptions {
    EncodeOptions {
        max_output_bytes: 1 << 20,
        modular: ModularOptions {
            predictor: Predictor::Zero,
            rct: None,
            fixed_tree: Some(FixedTree::Zero),
            lz77: false,
            entropy: Coding::Prefix,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gray(pixels: PixelData) -> Image {
    Image::new(pixels.len() as u32, 1, PixelLayout::Gray, pixels).unwrap()
}

#[derive(Default)]
struct Sink {
    writes: usize,
    bytes: Vec<u8>,
    capacity: Option<usize>,
    interrupt_once: bool,
    write_zero: bool,
}

impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.writes += 1;
        if std::mem::take(&mut self.interrupt_once) {
            return Err(io::ErrorKind::Interrupted.into());
        }
        if self.write_zero {
            return Ok(0);
        }
        let remaining = self.capacity.unwrap_or(usize::MAX) - self.bytes.len();
        if remaining == 0 {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        let count = remaining.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn codec_error_without_writes(canvas: &Image, frames: &[Frame<'_>], opts: &EncodeOptions) {
    let mut sink = Sink::default();
    let error = rs_jxl::encode_frames_to(canvas, frames, opts, &mut sink).unwrap_err();
    assert!(
        !matches!(error, Error::Io(_)),
        "unexpected sink error: {error}"
    );
    assert_eq!(sink.writes, 0, "codec error must not touch Write");
    assert!(sink.bytes.is_empty());
}

#[test]
fn declared_integer_depth_and_float_storage_cannot_be_reinterpreted() {
    let integer = gray(PixelData::U16(vec![0, 4095]));
    let mut metadata = integer.metadata().clone();
    metadata.bit_depth = BitDepth::new(12).unwrap();
    let narrow = integer.clone().with_metadata(metadata, vec![]).unwrap();
    assert_eq!(narrow.pixels(), integer.pixels(), "no integer rescaling");
    let too_large = gray(PixelData::U16(vec![0, 4096]));
    assert!(
        too_large
            .with_metadata(narrow.metadata().clone(), vec![])
            .is_err()
    );

    let float = gray(PixelData::F16(vec![0, 0x3c00]));
    assert!(
        integer
            .clone()
            .with_metadata(float.metadata().clone(), vec![])
            .is_err()
    );
    assert!(
        float
            .with_metadata(integer.metadata().clone(), vec![])
            .is_err()
    );
    for (bits_per_sample, exponent_bits_per_sample) in [(0, 0), (32, 0), (255, 255), (16, 1)] {
        let mut metadata = integer.metadata().clone();
        metadata.bit_depth = BitDepth {
            bits_per_sample,
            exponent_bits_per_sample,
        };
        assert!(integer.clone().with_metadata(metadata, vec![]).is_err());
    }
}

#[test]
fn level5_axis_boundary_remains_encodable_subject_to_resource_limits() {
    for width in [262144, 262145] {
        let image = gray(PixelData::U8(vec![0; width]));
        let opts = EncodeOptions {
            // Stop at the image header, without traversing a long row.
            max_output_bytes: 1,
            ..options()
        };
        let error = rs_jxl::encode(&image, &opts).unwrap_err();
        assert!(
            matches!(error, Error::Limit(_)),
            "both valid Level 10 axes reach the output budget"
        );
        let bounded = EncodeOptions {
            max_pixels: width as u64 - 1,
            ..options()
        };
        assert!(matches!(
            rs_jxl::encode(&image, &bounded),
            Err(Error::Limit("pixel count"))
        ));
    }
}

#[test]
fn u31_adjacent_samples_remain_distinct_even_when_f32_oracle_cannot_distinguish_them() {
    let a = (1_u32 << 30) + 1;
    let b = a + 1;
    assert_eq!(
        a as f32, b as f32,
        "fixture must expose oracle precision loss"
    );
    let a = gray(PixelData::U32(vec![a, i32::MAX as u32]));
    let b = gray(PixelData::U32(vec![b, i32::MAX as u32]));
    assert_eq!(
        multiframe::difference_bounds(&a, &b).unwrap(),
        Some(Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1
        })
    );
    // Acceptance of a stream is not independent evidence of U31 recovery.
    assert!(rs_jxl::encode(&a, &options()).is_ok());
    assert!(rs_jxl::encode(&b, &options()).is_ok());
}

#[test]
fn mixed_extra_storage_has_independent_depth_and_subsampled_lengths() {
    let base = Image::new(
        3,
        1,
        PixelLayout::Rgba,
        PixelData::U8([7, 11, 13, 0].repeat(3)),
    )
    .unwrap();
    let mut metadata = base.metadata().clone();
    let mut depth = ExtraChannelInfo::new(ExtraChannelType::Depth, 12).unwrap();
    depth.dim_shift = 1;
    let mut selection = ExtraChannelInfo::new(ExtraChannelType::SelectionMask, 8).unwrap();
    selection.bit_depth = BitDepth::floating(32, 8).unwrap();
    metadata.extra_channels.extend([depth, selection]);
    let planes = vec![
        PixelData::U16(vec![4095, 1]),
        PixelData::F32(vec![-0.0, 1.0, f32::from_bits(0x7fc0_0123)]),
    ];
    let image = base
        .clone()
        .with_metadata(metadata.clone(), planes.clone())
        .unwrap();
    assert_eq!(image.channel_dimensions(4), (2, 1));
    assert_eq!(image.channel_depth(5), BitDepth::floating(32, 8).unwrap());
    assert_eq!(image.pixels(), base.pixels());
    assert_eq!(image.extra_planes(), planes);
    assert!(rs_jxl::encode(&image, &options()).is_ok());
    for bad in [
        PixelData::U16(vec![4095]),
        PixelData::U16(vec![4096, 1]),
        PixelData::F16(vec![0, 1]),
    ] {
        assert!(
            base.clone()
                .with_metadata(metadata.clone(), vec![bad, planes[1].clone()])
                .is_err()
        );
    }
    metadata.extra_channels[1].dim_shift = 255;
    assert!(
        base.with_metadata(metadata, planes).is_err(),
        "reject before shifting by 255"
    );
}

#[test]
fn metadata_extra_count_is_not_the_encoder_level10_limit() {
    for count in [256, 257] {
        let base = gray(PixelData::U8(vec![37]));
        let mut metadata = base.metadata().clone();
        metadata.extra_channels =
            vec![ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap(); count];
        let image = base
            .with_metadata(metadata, vec![PixelData::U8(vec![3]); count])
            .unwrap();
        if count == 256 {
            assert!(rs_jxl::encode(&image, &options()).is_ok());
        } else {
            codec_error_without_writes(&image, &[Frame::new(&image)], &options());
        }
    }
}

#[test]
fn unencoded_canvas_and_reference_only_images_still_obey_pixel_limits() {
    let small = gray(PixelData::U8(vec![1]));
    let large = gray(PixelData::U8(vec![1, 2]));
    let opts = EncodeOptions {
        max_pixels: 1,
        ..options()
    };
    codec_error_without_writes(&large, &[Frame::new(&small)], &opts);
    let frames = [
        Frame {
            image: &large,
            options: FrameOptions {
                reference_only: true,
                save_before_color_transform: true,
                ..Default::default()
            },
        },
        Frame::new(&small),
    ];
    codec_error_without_writes(&small, &frames, &opts);
    // max_pixels is per image, not a cumulative sequence-work ceiling.
    let frames = [Frame::new(&small), Frame::new(&small), Frame::new(&small)];
    assert!(rs_jxl::encode_frames(&small, &frames, &opts).is_ok());
}

#[test]
fn aggregate_output_limit_and_late_metadata_error_never_touch_sink() {
    let image = gray(PixelData::U8(vec![1, 19, 255]));
    let frames = [Frame::new(&image), Frame::new(&image), Frame::new(&image)];
    let bytes = rs_jxl::encode_frames(&image, &frames, &options()).unwrap();
    let mut opts = options();
    opts.max_output_bytes = bytes.len();
    assert!(rs_jxl::encode_frames(&image, &frames, &opts).is_ok());
    opts.max_output_bytes -= 1;
    codec_error_without_writes(&image, &frames, &opts);

    let mut metadata = image.metadata().clone();
    metadata.tone_mapping.intensity_target = 1000.0;
    let incompatible = image.clone().with_metadata(metadata, vec![]).unwrap();
    codec_error_without_writes(
        &image,
        &[Frame::new(&image), Frame::new(&incompatible)],
        &options(),
    );
}

#[test]
fn write_sink_retries_interruption_but_reports_partial_io_and_write_zero() {
    let image = gray(PixelData::U8(vec![0, 17, 255]));
    let mut sink = Sink {
        interrupt_once: true,
        ..Default::default()
    };
    rs_jxl::encode_to(&image, &options(), &mut sink).unwrap();
    assert!(sink.writes >= 2);
    assert!(!sink.bytes.is_empty());
    for write_zero in [false, true] {
        let mut sink = Sink {
            capacity: Some(3),
            write_zero,
            ..Default::default()
        };
        let Error::Io(error) = rs_jxl::encode_to(&image, &options(), &mut sink).unwrap_err() else {
            panic!("expected I/O error");
        };
        assert_eq!(
            error.kind(),
            if write_zero {
                io::ErrorKind::WriteZero
            } else {
                io::ErrorKind::BrokenPipe
            }
        );
        assert_eq!(sink.bytes.len(), if write_zero { 0 } else { 3 });
    }
}

#[test]
fn names_are_byte_bounded_and_maximum_animation_fields_are_representable() {
    let base = gray(PixelData::U8(vec![37]));
    let mut metadata = base.metadata().clone();
    metadata.animation = Some(Animation {
        ticks_per_second_numerator: 1 << 30,
        ticks_per_second_denominator: 1024,
        loops: u32::MAX,
        timecodes: true,
    });
    let canvas = base.clone().with_metadata(metadata, vec![]).unwrap();
    let mut frame = Frame {
        image: &base,
        options: FrameOptions {
            duration: u32::MAX,
            timecode: u32::MAX,
            name: "\u{4e2d}".repeat(357),
            ..Default::default()
        },
    };
    assert_eq!(frame.options.name.len(), 1071);
    assert!(rs_jxl::encode_frames(&canvas, &[frame], &options()).is_ok());
    frame = Frame {
        image: &base,
        options: FrameOptions {
            name: "\u{4e2d}".repeat(358),
            ..Default::default()
        },
    };
    codec_error_without_writes(&canvas, &[frame], &options());
    for (numerator, denominator) in [((1 << 30) + 1, 1), (1, 1025)] {
        assert!(
            Animation {
                ticks_per_second_numerator: numerator,
                ticks_per_second_denominator: denominator,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}

#[test]
fn crop_arithmetic_and_unrepresentable_frame_origins_return_errors() {
    let image = gray(PixelData::U8(vec![1, 2]));
    for rect in [
        Rect {
            x: u32::MAX,
            y: 0,
            width: 2,
            height: 1,
        },
        Rect {
            x: 0,
            y: u32::MAX,
            width: 1,
            height: 2,
        },
        Rect {
            x: 0,
            y: 0,
            width: u32::MAX,
            height: 1,
        },
    ] {
        assert!(multiframe::crop(&image, rect).is_err());
    }
    for origin in [(i32::MIN, 0), (i32::MAX, 0), (0, i32::MIN), (0, i32::MAX)] {
        let frames = [
            Frame::new(&image),
            Frame {
                image: &image,
                options: FrameOptions {
                    origin,
                    ..Default::default()
                },
            },
        ];
        codec_error_without_writes(&image, &frames, &options());
    }
}

#[test]
fn zero_duration_crop_validates_each_extra_reference_and_metadata() {
    let base = gray(PixelData::U8(vec![11, 23]));
    let mut metadata = base.metadata().clone();
    let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
    extra.name = "depth".into();
    metadata.extra_channels.push(extra);
    let full = base
        .with_metadata(metadata, vec![PixelData::U16(vec![257, 65535])])
        .unwrap();
    let crop = multiframe::crop(
        &full,
        Rect {
            x: 1,
            y: 0,
            width: 1,
            height: 1,
        },
    )
    .unwrap();
    let make = |source| {
        [
            Frame {
                image: &full,
                options: FrameOptions {
                    reference_only: true,
                    save_as_reference: 1,
                    ..Default::default()
                },
            },
            Frame {
                image: &crop,
                options: FrameOptions {
                    origin: (1, 0),
                    blend: BlendInfo {
                        source: 1,
                        ..Default::default()
                    },
                    extra_blend: vec![BlendInfo {
                        source,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            },
        ]
    };
    for source in 0..4 {
        let bytes = rs_jxl::encode_frames(&full, &make(source), &options()).unwrap();
        let decoded = rs_jxl::decode::decode(&bytes, &Default::default()).unwrap();
        assert_eq!(decoded.frames.len(), 1);
        assert_eq!(decoded.frames[0].pixels, PixelData::U8(vec![11, 23]));
        // Color uses saved slot 1; each extra independently selects its source.
        // Unsaved slots are valid zero backgrounds, not missing references.
        assert_eq!(
            decoded.frames[0].extra_planes,
            vec![PixelData::U16(vec![
                if source == 1 { 257 } else { 0 },
                65535
            ])],
            "extra source={source}"
        );
    }
    codec_error_without_writes(&full, &make(4), &options());
    let mut changed = crop.metadata().clone();
    changed.extra_channels[0].name = "different-depth-name".into();
    let different = crop
        .clone()
        .with_metadata(changed, crop.extra_planes().to_vec())
        .unwrap();
    let mut frames = make(1);
    frames[1].image = &different;
    codec_error_without_writes(&full, &frames, &options());
    frames[1].image = &crop;
    frames[1].options.blend = BlendInfo {
        mode: BlendMode::Blend,
        source: 1,
        alpha_channel: 1,
        ..Default::default()
    };
    codec_error_without_writes(&full, &frames, &options());
}

#[test]
fn legacy_sample_limit_rejects_instead_of_silently_truncating() {
    let image = gray(PixelData::U8((0..64).collect()));
    let mut opts = options();
    opts.modular.fixed_tree = None;
    opts.modular.tree_samples = 64;
    opts.modular.max_tree_leaves = 1;
    opts.modular.sample_limit = Some(1);
    assert!(matches!(
        rs_jxl::encode(&image, &opts),
        Err(Error::Limit(_))
    ));
    codec_error_without_writes(&image, &[Frame::new(&image)], &opts);
    opts.modular.sample_limit = Some(64);
    assert!(rs_jxl::encode(&image, &opts).is_ok());
    // This does NOT prove pre-allocation enforcement. PA-03's original collector
    // created the complete batch before checking the cap; audit that separately.
}

#[test]
#[cfg(target_pointer_width = "32")]
fn legacy_sample_budget_rejection_must_not_overflow_on_32_bit_targets() {
    // 64 KiB input. At next=32768, the legacy midpoint product
    // (2 * next + 1) * 65536 exceeds u32 before the sample cap is checked.
    let image = Image::new(
        256,
        256,
        PixelLayout::Gray,
        PixelData::U8((0..65536).map(|i| i as u8).collect()),
    )
    .unwrap();
    let mut opts = options();
    opts.modular.fixed_tree = None;
    opts.modular.tree_samples = 65536;
    opts.modular.sample_limit = Some(1);
    assert!(matches!(
        rs_jxl::encode(&image, &opts),
        Err(Error::Limit(_))
    ));
}

#[test]
fn equivalent_f16_and_f32_storage_has_no_difference_bounds() {
    // Known regression PA-01: compare declared samples, not container words.
    let half = gray(PixelData::F16(vec![0x3c00, 0x8000]));
    let single = gray(PixelData::F32(vec![1.0, -0.0]))
        .with_metadata(half.metadata().clone(), vec![])
        .unwrap();
    assert_eq!(multiframe::difference_bounds(&half, &single).unwrap(), None);

    let base = gray(PixelData::U8(vec![1, 2]));
    let mut metadata = base.metadata().clone();
    let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
    extra.bit_depth = BitDepth::floating(16, 5).unwrap();
    metadata.extra_channels.push(extra);
    let half = base
        .clone()
        .with_metadata(metadata.clone(), vec![half.pixels().clone()])
        .unwrap();
    let single = base
        .with_metadata(metadata, vec![single.pixels().clone()])
        .unwrap();
    assert_eq!(multiframe::difference_bounds(&half, &single).unwrap(), None);
}

#[test]
fn malformed_patch_limits_are_rejected_even_when_sample_depth_is_ineligible() {
    // Known regression PA-02: depth gating bypasses option validation.
    let integer = gray(PixelData::U8(vec![1]));
    let wide = gray(PixelData::U32(vec![1]));
    let float = gray(PixelData::F32(vec![1.0]));
    for source in [false, true] {
        let mut opts = options();
        let limits = PatchOptions {
            max_patch_size: 0,
            ..Default::default()
        };
        if source {
            opts.source_patches = Some(SourcePatchConfig {
                limits,
                ..Default::default()
            });
        } else {
            opts.patches = Some(limits);
        }
        for image in [&integer, &wide, &float] {
            codec_error_without_writes(image, &[Frame::new(image)], &opts);
        }
    }
}
