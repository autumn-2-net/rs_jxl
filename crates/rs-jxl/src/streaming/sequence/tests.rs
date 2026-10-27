use super::*;
use crate::frame::{BlendInfo, BlendMode};
use crate::metadata::{Animation, BitDepth, ExtraChannelInfo, ExtraChannelType};
use crate::region_input::{ImageRegionSource, RegionLease, RegionRequest};
use crate::{Image, PixelData, PixelLayout};
use std::cell::{Cell, RefCell};
use std::io::{Cursor, SeekFrom};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::thread::{self, ThreadId};

fn gray(width: u32, height: u32) -> Image {
    Image::new(
        width,
        height,
        PixelLayout::Gray,
        PixelData::U8((0..width * height).map(|i| (i % 251) as u8).collect()),
    )
    .unwrap()
}

fn options(threads: usize) -> EncodeOptions {
    let mut options = EncodeOptions {
        threads,
        group_size_shift: 0,
        ..Default::default()
    };
    options.modular.palette_policy = Some(crate::modular::palette::PaletteOptions {
        groupwise: true,
        palette_colors: 0,
        channel_colors_pre_transform_percent: 0.0,
        channel_colors_percent: 0.0,
        ..Default::default()
    });
    options
}

fn config(threads: usize) -> FrameEncoderOptions {
    FrameEncoderOptions {
        threads,
        buffering: FrameBuffering::CurrentFrame {
            max_sample_bytes: 1024 * 1024,
        },
        ..Default::default()
    }
}

struct LocalSource<'a> {
    inner: ImageRegionSource<'a>,
    calls: Rc<Cell<usize>>,
    thread: ThreadId,
    panic: bool,
}
impl RegionSource for LocalSource<'_> {
    fn descriptor(&self) -> &ImageDescriptor {
        self.inner.descriptor()
    }
    fn acquire(&mut self, request: RegionRequest) -> Result<RegionLease<'_>> {
        assert_eq!(thread::current().id(), self.thread);
        self.calls.set(self.calls.get() + 1);
        assert!(!self.panic, "injected input panic");
        self.inner.acquire(request)
    }
}

#[derive(Clone)]
struct LocalSink(Rc<RefCell<Cursor<Vec<u8>>>>, ThreadId);
impl Write for LocalSink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        assert_eq!(thread::current().id(), self.1);
        self.0.borrow_mut().write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        panic!("must not flush")
    }
}
impl Seek for LocalSink {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        assert_eq!(thread::current().id(), self.1);
        self.0.borrow_mut().seek(position)
    }
}

fn source(image: &Image) -> LocalSource<'_> {
    LocalSource {
        inner: ImageRegionSource::new(image).unwrap(),
        calls: Rc::new(Cell::new(0)),
        thread: thread::current().id(),
        panic: false,
    }
}

fn sink() -> LocalSink {
    LocalSink(
        Rc::new(RefCell::new(Cursor::new(Vec::new()))),
        thread::current().id(),
    )
}

fn bare_codestream(bytes: &[u8]) -> Vec<u8> {
    if bytes.starts_with(&[0xff, 0x0a]) {
        return bytes.to_vec();
    }
    let mut result = Vec::new();
    let mut offset = 0;
    let mut chunks = 0;
    let mut complete = false;
    while offset < bytes.len() {
        let mut size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let kind = &bytes[offset + 4..offset + 8];
        let mut header = 8;
        if size == 1 {
            size = u64::from_be_bytes(bytes[offset + 8..offset + 16].try_into().unwrap()) as usize;
            header = 16;
        } else if size == 0 {
            size = bytes.len() - offset;
        }
        if kind == b"jxlp" {
            assert!(!complete);
            let counter = u32::from_be_bytes(
                bytes[offset + header..offset + header + 4]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(counter & 0x7fff_ffff, chunks);
            complete = counter >> 31 != 0;
            chunks += 1;
            result.extend_from_slice(&bytes[offset + header + 4..offset + size]);
        } else if kind == b"jxlc" {
            result.extend_from_slice(&bytes[offset + header..offset + size]);
        }
        offset += size;
    }
    assert!(chunks == 0 || complete);
    result
}

#[test]
fn sequential_area_frames_and_crop_use_non_send_callbacks_and_one_pool() {
    for threads in [1, 3] {
        let image = gray(1030, 3);
        let crop = gray(1025, 3);
        let output = sink();
        let shared = output.0.clone();
        let mut encoder = FrameEncoder::new(
            ImageDescriptor::from_image(&image).unwrap(),
            FrameEncoderOptions {
                threads,
                ..Default::default()
            },
            output,
        )
        .unwrap();
        let first = encoder
            .write_frame(
                &mut source(&image),
                &FrameOptions::default(),
                &options(threads),
                false,
            )
            .unwrap();
        assert_eq!(first.mode, FrameProgressMode::FrameAreas);
        let prefix = shared.borrow().get_ref().clone();
        assert_eq!(prefix.len(), first.finalized_bytes);
        let last = encoder
            .write_frame(
                &mut source(&crop),
                &FrameOptions {
                    origin: (3, 0),
                    name: "crop".into(),
                    ..Default::default()
                },
                &options(threads),
                true,
            )
            .unwrap();
        assert_eq!(last.frame_index, 1);
        assert_eq!(last.mode, FrameProgressMode::FrameAreas);
        assert_eq!(&shared.borrow().get_ref()[..prefix.len()], prefix);
        assert!(
            encoder
                .write_frame(
                    &mut source(&image),
                    &FrameOptions::default(),
                    &options(threads),
                    true
                )
                .is_err()
        );
        let (_, length) = encoder.finish().unwrap();
        assert_eq!(length, shared.borrow().get_ref().len());
        assert_eq!(length, last.finalized_bytes);
        assert!(!bare_codestream(shared.borrow().get_ref()).is_empty());
    }
}

#[test]
fn buffered_sequence_matches_existing_frame_encoder_and_can_release_each_input() {
    let base = gray(7, 5);
    let mut metadata = base.metadata().clone();
    metadata.animation = Some(Animation {
        timecodes: true,
        ..Default::default()
    });
    let base = base.with_metadata(metadata.clone(), vec![]).unwrap();
    let crop = gray(3, 2).with_metadata(metadata, vec![]).unwrap();
    let settings = [
        FrameOptions {
            reference_only: true,
            save_as_reference: 1,
            ..Default::default()
        },
        FrameOptions {
            duration: 0,
            save_as_reference: 2,
            ..Default::default()
        },
        FrameOptions {
            origin: (2, 1),
            duration: 4,
            timecode: 17,
            name: "last".into(),
            blend: BlendInfo {
                source: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    ];
    let images = [&base, &base, &crop];
    let options = options(2);
    let frames: Vec<_> = images
        .iter()
        .zip(&settings)
        .map(|(&image, options)| Frame {
            image,
            options: options.clone(),
        })
        .collect();
    let expected = crate::encode_frames(&base, &frames, &options).unwrap();
    let mut encoder = FrameEncoder::new(
        ImageDescriptor::from_image(&base).unwrap(),
        config(2),
        Cursor::new(Vec::new()),
    )
    .unwrap();
    for (i, (&image, frame)) in images.iter().zip(&settings).enumerate() {
        let progress = encoder
            .write_frame(&mut source(image), frame, &options, i == 2)
            .unwrap();
        assert_eq!(progress.mode, FrameProgressMode::CurrentFrameBuffered);
    }
    let (output, _) = encoder.finish().unwrap();
    assert_eq!(
        bare_codestream(output.get_ref()),
        bare_codestream(&expected)
    );
}

#[test]
fn rejection_before_callbacks_is_retryable_and_limits_are_stream_wide() {
    let image = gray(7, 5);
    let descriptor = ImageDescriptor::from_image(&image).unwrap();
    let mut input = source(&image);
    let output = sink();
    let shared = output.0.clone();
    let mut encoder =
        FrameEncoder::new(descriptor.clone(), FrameEncoderOptions::default(), output).unwrap();
    assert!(
        encoder
            .write_frame(&mut input, &FrameOptions::default(), &options(1), true)
            .is_err()
    );
    assert_eq!(input.calls.get(), 0);
    assert!(shared.borrow().get_ref().is_empty());
    assert!(!encoder.is_poisoned());
    assert!(encoder.finish().is_err());
    let mut bounded = config(1);
    bounded.buffering = FrameBuffering::CurrentFrame {
        max_sample_bytes: 7 * 5 * 8 - 1,
    };
    let mut encoder = FrameEncoder::new(descriptor.clone(), bounded, sink()).unwrap();
    assert!(
        encoder
            .write_frame(&mut input, &FrameOptions::default(), &options(1), true)
            .is_err()
    );
    assert_eq!(input.calls.get(), 0);
    let mut encoder =
        FrameEncoder::new(descriptor.clone(), config(1), Cursor::new(Vec::new())).unwrap();
    let invalid = FrameOptions {
        blend: BlendInfo {
            mode: BlendMode::Add,
            source: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(
        encoder
            .write_frame(&mut input, &invalid, &options(1), false)
            .is_err()
    );
    assert_eq!(input.calls.get(), 0);
    encoder
        .write_frame(&mut input, &FrameOptions::default(), &options(1), false)
        .unwrap();
    encoder
        .write_frame(&mut input, &FrameOptions::default(), &options(1), true)
        .unwrap();
    let (_, length) = encoder.finish().unwrap();
    for maximum in [length, length - 1] {
        let mut config = config(1);
        config.max_output_bytes = maximum;
        let output = sink();
        let shared = output.0.clone();
        let mut encoder = FrameEncoder::new(descriptor.clone(), config, output).unwrap();
        encoder
            .write_frame(&mut input, &FrameOptions::default(), &options(1), false)
            .unwrap();
        let result = encoder.write_frame(&mut input, &FrameOptions::default(), &options(1), true);
        assert_eq!(result.is_ok(), maximum == length);
        assert!(shared.borrow().get_ref().len() <= maximum);
        if result.is_ok() {
            encoder.finish().unwrap();
        }
    }
}

#[test]
fn input_unwind_poison_preserves_finalized_prefix() {
    for width in [7, 1025] {
        let image = gray(width, 3);
        let output = sink();
        let shared = output.0.clone();
        let mut encoder = FrameEncoder::new(
            ImageDescriptor::from_image(&image).unwrap(),
            config(1),
            output,
        )
        .unwrap();
        let first = encoder
            .write_frame(
                &mut source(&image),
                &FrameOptions::default(),
                &options(1),
                false,
            )
            .unwrap();
        let prefix = shared.borrow().get_ref().clone();
        let mut input = source(&image);
        input.panic = true;
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _ =
                    encoder.write_frame(&mut input, &FrameOptions::default(), &options(1), true);
            }))
            .is_err()
        );
        assert!(encoder.is_poisoned());
        assert_eq!(encoder.finalized_bytes(), first.finalized_bytes);
        assert_eq!(&shared.borrow().get_ref()[..prefix.len()], prefix);
        assert!(
            encoder
                .write_frame(
                    &mut source(&image),
                    &FrameOptions::default(),
                    &options(1),
                    true
                )
                .is_err()
        );
        assert!(encoder.finish().is_err());
    }
}

#[test]
fn semantic_metadata_allows_separate_alpha_and_preserves_subsampled_extras() {
    let rgba = Image::new(
        3,
        2,
        PixelLayout::Rgba,
        PixelData::U8([12, 7, 98, 0, 42, 9, 3, 255].repeat(3)),
    )
    .unwrap();
    let mut metadata = rgba.metadata().clone();
    let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
    extra.dim_shift = 1;
    metadata.extra_channels.push(extra);
    let rgba = rgba
        .with_metadata(metadata.clone(), vec![PixelData::U16(vec![1000, 65535])])
        .unwrap();
    let rgb = Image::new(
        3,
        2,
        PixelLayout::Rgb,
        PixelData::U8([12, 7, 98, 42, 9, 3].repeat(3)),
    )
    .unwrap()
    .with_metadata(
        metadata,
        vec![
            PixelData::U8([0, 255].repeat(3)),
            PixelData::U16(vec![1000, 65535]),
        ],
    )
    .unwrap();
    let options = options(1);
    let expected =
        crate::encode_frames(&rgba, &[Frame::new(&rgba), Frame::new(&rgba)], &options).unwrap();
    let mut encoder = FrameEncoder::new(
        ImageDescriptor::from_image(&rgba).unwrap(),
        config(1),
        Cursor::new(Vec::new()),
    )
    .unwrap();
    encoder
        .write_frame(
            &mut source(&rgba),
            &FrameOptions::default(),
            &options,
            false,
        )
        .unwrap();
    encoder
        .write_frame(&mut source(&rgb), &FrameOptions::default(), &options, true)
        .unwrap();
    let (output, _) = encoder.finish().unwrap();
    assert_eq!(
        bare_codestream(output.get_ref()),
        bare_codestream(&expected)
    );
}

fn assert_buffered_words(image: &Image) {
    let descriptor = ImageDescriptor::from_image(image).unwrap();
    let bytes = region_input::buffered_sample_bytes(&descriptor).unwrap();
    let actual = region_input::read_frame(&descriptor, &mut source(image), bytes).unwrap();
    assert_eq!(actual.metadata(), image.metadata());
    for channel in 0..image.channel_count() {
        let (width, height) = image.channel_dimensions(channel);
        assert_eq!(actual.channel_dimensions(channel), (width, height));
        for sample in 0..width * height {
            assert_eq!(
                actual.modular_sample(channel, sample),
                image.modular_sample(channel, sample),
                "channel={channel}, sample={sample}"
            );
        }
    }
}

#[test]
fn fallback_all_binary16_words_and_mixed_integer_extras_are_exact() {
    let image = Image::new(
        256,
        256,
        PixelLayout::Gray,
        PixelData::F16((0..=u16::MAX).collect()),
    )
    .unwrap();
    assert_buffered_words(&image);
    let image = gray(17, 9);
    let mut metadata = image.metadata().clone();
    let mut extras = Vec::new();
    for (bits, shift) in [(1, 0), (12, 1), (16, 2), (24, 3), (31, 3)] {
        let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, bits).unwrap();
        extra.dim_shift = shift;
        metadata.extra_channels.push(extra);
        let count = 17usize.div_ceil(1 << shift) * 9usize.div_ceil(1 << shift);
        let maximum = (1u32 << bits) - 1;
        extras.push(PixelData::U32(
            (0..count).map(|i| maximum - (i as u32 & maximum)).collect(),
        ));
    }
    let mut invalid = metadata.clone();
    invalid.extra_channels.last_mut().unwrap().dim_shift = 4;
    assert!(matches!(
        ImageDescriptor::new(17, 9, PixelLayout::Gray, invalid),
        Err(Error::InvalidInput(_))
    ));
    assert_buffered_words(&image.with_metadata(metadata, extras).unwrap());
}

#[test]
fn fallback_special_float_sequence_and_terminal_zero_layer_match_existing() {
    let half = [
        0, 0x8000, 1, 0x3ff, 0x400, 0x3c00, 0x7c00, 0xfc00, 0x7e01, 0xfe55, 0x7c01, 0xfc01, 0x7fff,
        0xffff,
    ];
    let single = [
        0,
        0x8000_0000,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0123,
        0xffc0_0055,
        0x7f80_0001,
        0xff80_0001,
    ];
    for pixels in [
        PixelData::F16(half.to_vec()),
        PixelData::F32(single.into_iter().map(f32::from_bits).collect()),
    ] {
        let image = Image::new(pixels.len() as u32, 1, PixelLayout::Gray, pixels).unwrap();
        assert_buffered_words(&image);
        let options = options(1);
        let expected =
            crate::encode_frames(&image, &[Frame::new(&image), Frame::new(&image)], &options)
                .unwrap();
        let mut encoder = FrameEncoder::new(
            ImageDescriptor::from_image(&image).unwrap(),
            config(1),
            Cursor::new(Vec::new()),
        )
        .unwrap();
        encoder
            .write_frame(
                &mut source(&image),
                &FrameOptions::default(),
                &options,
                false,
            )
            .unwrap();
        encoder
            .write_frame(
                &mut source(&image),
                &FrameOptions::default(),
                &options,
                true,
            )
            .unwrap();
        let (output, _) = encoder.finish().unwrap();
        assert_eq!(
            bare_codestream(output.get_ref()),
            bare_codestream(&expected)
        );
    }
}

#[test]
fn index_and_metadata_are_bounded_and_offsets_exclude_boxes() {
    let image = gray(7, 5);
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation {
        ticks_per_second_numerator: 24,
        ..Default::default()
    });
    let image = image.with_metadata(metadata, vec![]).unwrap();
    let options = options(1);
    let frames: Vec<_> = [11, 0, 33, 0]
        .into_iter()
        .map(|duration| Frame {
            image: &image,
            options: FrameOptions {
                duration,
                ..Default::default()
            },
        })
        .collect();
    let mut expected = Vec::new();
    crate::streaming::encode_indexed_frames_streaming_to(
        &image,
        &frames,
        &[0, 2, 3],
        &options,
        &mut expected,
    )
    .unwrap();
    let mut config = config(1);
    config.index = Some(FrameIndexOptions {
        max_entries: 3,
        max_payload_bytes: 100,
    });
    let output = sink();
    let shared = output.0.clone();
    let mut encoder =
        FrameEncoder::new(ImageDescriptor::from_image(&image).unwrap(), config, output).unwrap();
    assert!(encoder.write_metadata(*b"brob", &[]).is_err());
    assert!(encoder.write_metadata(*b"Exif", &[0; 3]).is_err());
    assert!(encoder.write_metadata(*b"jxli", &[]).is_err());
    encoder.write_metadata(*b"xml ", b"<before/>").unwrap();
    let prefix = shared.borrow().get_ref().clone();
    assert!(
        encoder
            .write_frame(&mut source(&image), &frames[0].options, &options, false)
            .is_err()
    );
    assert_eq!(*shared.borrow().get_ref(), prefix);
    for (i, frame) in frames.iter().enumerate() {
        if i == 1 {
            encoder
                .write_frame(&mut source(&image), &frame.options, &options, false)
                .unwrap();
        } else {
            encoder
                .write_indexed_frame(&mut source(&image), &frame.options, &options, i == 3)
                .unwrap();
        }
        encoder.write_metadata(*b"jumb", &[i as u8]).unwrap();
    }
    assert_eq!(
        encoder.index.as_ref().unwrap().payload().unwrap(),
        index_payload(&expected)
    );
    encoder.write_metadata(*b"Exif", &[0; 4]).unwrap();
    let (_, size) = encoder.finish().unwrap();
    assert_eq!(size, shared.borrow().get_ref().len());
    assert_eq!(
        bare_codestream(shared.borrow().get_ref()),
        bare_codestream(&expected)
    );
    assert_eq!(
        index_payload(shared.borrow().get_ref()),
        index_payload(&expected)
    );
}

fn index_payload(bytes: &[u8]) -> Vec<u8> {
    let mut offset = 0;
    while offset < bytes.len() {
        let mut size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let mut header = 8;
        if size == 1 {
            header = 16;
            size = u64::from_be_bytes(bytes[offset + 8..offset + 16].try_into().unwrap()) as usize;
        }
        if &bytes[offset + 4..offset + 8] == b"jxli" {
            return bytes[offset + header..offset + size].to_vec();
        }
        offset += size;
    }
    panic!("missing index")
}

#[test]
fn index_rejects_cross_interval_reference_and_entry_cap_without_input() {
    let image = gray(7, 5);
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation::default());
    let image = image.with_metadata(metadata, vec![]).unwrap();
    let options = options(1);
    let mut config = config(1);
    config.index = Some(FrameIndexOptions {
        max_entries: 2,
        max_payload_bytes: 100,
    });
    let mut encoder =
        FrameEncoder::new(ImageDescriptor::from_image(&image).unwrap(), config, sink()).unwrap();
    encoder
        .write_indexed_frame(
            &mut source(&image),
            &FrameOptions {
                duration: 1,
                save_as_reference: 1,
                ..Default::default()
            },
            &options,
            false,
        )
        .unwrap();
    encoder
        .write_indexed_frame(
            &mut source(&image),
            &FrameOptions {
                duration: 1,
                ..Default::default()
            },
            &options,
            false,
        )
        .unwrap();
    let mut input = source(&image);
    assert!(
        encoder
            .write_frame(
                &mut input,
                &FrameOptions {
                    blend: BlendInfo {
                        mode: BlendMode::Add,
                        source: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                &options,
                true
            )
            .is_err()
    );
    assert!(
        encoder
            .write_indexed_frame(&mut input, &FrameOptions::default(), &options, true)
            .is_err()
    );
    assert_eq!(input.calls.get(), 0);
    assert!(!encoder.is_poisoned());
    encoder
        .write_frame(&mut input, &FrameOptions::default(), &options, true)
        .unwrap();
    encoder.finish().unwrap();
}

#[test]
fn metadata_and_reference_stage_validation_precede_acquisition() {
    let image = gray(7, 5);
    let mut encoder = FrameEncoder::new(
        ImageDescriptor::from_image(&image).unwrap(),
        config(1),
        sink(),
    )
    .unwrap();
    let mut input = source(&image);
    assert!(
        encoder
            .write_frame(
                &mut input,
                &FrameOptions {
                    duration: 1,
                    ..Default::default()
                },
                &options(1),
                false
            )
            .is_err()
    );
    assert!(
        encoder
            .write_frame(&mut input, &FrameOptions::default(), &options(2), false)
            .is_err()
    );
    assert_eq!(input.calls.get(), 0);
    encoder
        .write_frame(
            &mut input,
            &FrameOptions {
                reference_only: true,
                save_as_reference: 1,
                save_before_color_transform: true,
                ..Default::default()
            },
            &options(1),
            false,
        )
        .unwrap();
    let calls = input.calls.get();
    assert!(
        encoder
            .write_frame(
                &mut input,
                &FrameOptions {
                    blend: BlendInfo {
                        mode: BlendMode::Add,
                        source: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                &options(1),
                true
            )
            .is_err()
    );
    let other = Image::new(7, 5, PixelLayout::Gray, PixelData::U16(vec![1; 35])).unwrap();
    let mut other = source(&other);
    assert!(
        encoder
            .write_frame(&mut other, &FrameOptions::default(), &options(1), true)
            .is_err()
    );
    assert_eq!(other.calls.get(), 0);
    assert_eq!(input.calls.get(), calls);
    encoder
        .write_frame(&mut input, &FrameOptions::default(), &options(1), true)
        .unwrap();
    encoder.finish().unwrap();
}

#[test]
fn partial_output_error_or_unwind_poisons_without_changing_finalized_prefix() {
    struct FaultSink {
        inner: LocalSink,
        failure: Rc<Cell<u8>>,
    }
    impl Write for FaultSink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.failure.get() != 0 && !bytes.is_empty() {
                self.inner.write_all(&bytes[..1])?;
                assert_ne!(self.failure.get(), 2, "injected output panic");
                return Err(std::io::Error::other("injected partial output error"));
            }
            self.inner.write(bytes)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            panic!("must not flush")
        }
    }
    impl Seek for FaultSink {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(position)
        }
    }
    for failure in [1, 2] {
        let image = gray(7, 5);
        let output = sink();
        let shared = output.0.clone();
        let fail = Rc::new(Cell::new(0));
        let output = FaultSink {
            inner: output,
            failure: fail.clone(),
        };
        let mut encoder = FrameEncoder::new(
            ImageDescriptor::from_image(&image).unwrap(),
            config(1),
            output,
        )
        .unwrap();
        encoder
            .write_frame(
                &mut source(&image),
                &FrameOptions::default(),
                &options(1),
                false,
            )
            .unwrap();
        let prefix = shared.borrow().get_ref().clone();
        fail.set(failure);
        let result = catch_unwind(AssertUnwindSafe(|| {
            encoder.write_frame(
                &mut source(&image),
                &FrameOptions::default(),
                &options(1),
                true,
            )
        }));
        if failure == 1 {
            assert!(result.unwrap().is_err());
        } else {
            assert!(result.is_err());
        }
        assert!(encoder.is_poisoned());
        assert_eq!(encoder.finalized_bytes(), prefix.len());
        assert_eq!(&shared.borrow().get_ref()[..prefix.len()], prefix);
        assert!(encoder.write_metadata(*b"xml ", b"<x/>").is_err());
        assert!(encoder.finish().is_err());
    }
}

#[test]
fn index_payload_limit_and_total_output_reservation_precede_mutation() {
    let image = gray(7, 5);
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation::default());
    let image = image.with_metadata(metadata, vec![]).unwrap();
    let descriptor = ImageDescriptor::from_image(&image).unwrap();
    let mut config = config(1);
    config.index = Some(FrameIndexOptions {
        max_entries: 2,
        max_payload_bytes: 12,
    });
    let mut encoder = FrameEncoder::new(descriptor.clone(), config.clone(), sink()).unwrap();
    let mut input = source(&image);
    assert!(
        encoder
            .write_indexed_frame(
                &mut input,
                &FrameOptions {
                    duration: u32::MAX,
                    ..Default::default()
                },
                &options(1),
                true
            )
            .is_err()
    );
    assert_eq!(input.calls.get(), 0);
    encoder
        .write_indexed_frame(&mut input, &FrameOptions::default(), &options(1), true)
        .unwrap();
    let (_, size) = encoder.finish().unwrap();
    for maximum in [size, size - 1] {
        config.max_output_bytes = maximum;
        let output = sink();
        let shared = output.0.clone();
        let mut encoder = FrameEncoder::new(descriptor.clone(), config.clone(), output).unwrap();
        let result =
            encoder.write_indexed_frame(&mut input, &FrameOptions::default(), &options(1), true);
        assert_eq!(result.is_ok(), maximum == size);
        assert!(shared.borrow().get_ref().len() <= maximum);
        if result.is_ok() {
            encoder.finish().unwrap();
        }
    }
}

#[derive(Clone, Copy)]
enum LateFault {
    SeekTo(u64),
    IndexPayload,
}

struct LateFaultSink {
    inner: LocalSink,
    fault: Rc<Cell<Option<LateFault>>>,
    triggered: Rc<Cell<bool>>,
    index_start: Rc<Cell<Option<usize>>>,
    panic: bool,
}

impl LateFaultSink {
    fn fail<T>(&self) -> std::io::Result<T> {
        self.triggered.set(true);
        assert!(!self.panic, "injected late output panic");
        Err(std::io::Error::other("injected late output error"))
    }
}

impl Write for LateFaultSink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let position = self.inner.0.borrow().position() as usize;
        if matches!(self.fault.get(), Some(LateFault::IndexPayload))
            && self
                .index_start
                .get()
                .is_some_and(|start| position == start + 16)
            && !bytes.is_empty()
        {
            self.inner.write_all(&bytes[..1])?;
            return self.fail();
        }
        if bytes == b"jxli" {
            self.index_start.set(Some(position - 4));
        }
        self.inner.write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        panic!("must not flush")
    }
}

impl Seek for LateFaultSink {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        // Exercise failure after the underlying sink has already moved.
        let actual = self.inner.seek(position)?;
        if matches!((self.fault.get(), position), (Some(LateFault::SeekTo(target)), SeekFrom::Start(requested)) if target == requested)
        {
            return self.fail();
        }
        Ok(actual)
    }
}

#[test]
fn late_toc_and_box_length_seek_error_or_unwind_preserves_previous_prefix() {
    for toc in [true, false] {
        for panic in [false, true] {
            let image = gray(1025, 3);
            let output = sink();
            let shared = output.0.clone();
            let fault = Rc::new(Cell::new(None));
            let triggered = Rc::new(Cell::new(false));
            let output = LateFaultSink {
                inner: output,
                fault: fault.clone(),
                triggered: triggered.clone(),
                index_start: Rc::new(Cell::new(None)),
                panic,
            };
            let mut encoder = FrameEncoder::new(
                ImageDescriptor::from_image(&image).unwrap(),
                FrameEncoderOptions::default(),
                output,
            )
            .unwrap();
            let options = options(1);
            encoder
                .write_frame(
                    &mut source(&image),
                    &FrameOptions::default(),
                    &options,
                    false,
                )
                .unwrap();
            let prefix = shared.borrow().get_ref().clone();
            // Subsequent frames have a 20-byte extended jxlp header and no image header.
            let target = prefix.len() + if toc { 20 } else { 8 };
            fault.set(Some(LateFault::SeekTo(target as u64)));
            let mut input = source(&image);
            let result = catch_unwind(AssertUnwindSafe(|| {
                encoder.write_frame(&mut input, &FrameOptions::default(), &options, true)
            }));
            if panic {
                assert!(result.is_err());
            } else {
                assert!(matches!(result.unwrap(), Err(Error::Io(_))));
            }
            assert!(triggered.get());
            assert!(input.calls.get() >= 2);
            assert_eq!(shared.borrow().position(), target as u64);
            assert!(shared.borrow().get_ref().len() > prefix.len() + 20);
            assert!(encoder.is_poisoned());
            assert_eq!(encoder.finalized_bytes(), prefix.len());
            assert_eq!(&shared.borrow().get_ref()[..prefix.len()], prefix);
            let failed = shared.borrow().get_ref().clone();
            let mut retry = source(&image);
            assert!(
                encoder
                    .write_frame(&mut retry, &FrameOptions::default(), &options, true)
                    .is_err()
            );
            assert_eq!(retry.calls.get(), 0);
            assert!(encoder.write_metadata(*b"xml ", b"<retry/>").is_err());
            assert!(encoder.finish().is_err());
            assert_eq!(*shared.borrow().get_ref(), failed);
        }
    }
}

#[test]
fn final_index_payload_failure_keeps_advanced_final_frame_prefix_immutable() {
    for panic in [false, true] {
        let image = gray(7, 5);
        let mut metadata = image.metadata().clone();
        metadata.animation = Some(Animation::default());
        let image = image.with_metadata(metadata, vec![]).unwrap();
        let mut config = config(1);
        config.index = Some(FrameIndexOptions::default());
        let output = sink();
        let shared = output.0.clone();
        let fault = Rc::new(Cell::new(None));
        let triggered = Rc::new(Cell::new(false));
        let index_start = Rc::new(Cell::new(None));
        let output = LateFaultSink {
            inner: output,
            fault: fault.clone(),
            triggered: triggered.clone(),
            index_start: index_start.clone(),
            panic,
        };
        let mut encoder =
            FrameEncoder::new(ImageDescriptor::from_image(&image).unwrap(), config, output)
                .unwrap();
        let options = options(1);
        let first = FrameOptions {
            duration: 1,
            ..Default::default()
        };
        encoder
            .write_indexed_frame(&mut source(&image), &first, &options, false)
            .unwrap();
        let prefix = shared.borrow().get_ref().clone();
        fault.set(Some(LateFault::IndexPayload));
        let result = catch_unwind(AssertUnwindSafe(|| {
            encoder.write_frame(
                &mut source(&image),
                &FrameOptions::default(),
                &options,
                true,
            )
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert!(matches!(result.unwrap(), Err(Error::Io(_))));
        }
        assert!(triggered.get());
        let finalized = encoder.finalized_bytes();
        assert!(finalized > prefix.len());
        assert_eq!(Some(finalized), index_start.get());
        let failed = shared.borrow().get_ref().clone();
        assert_eq!(&failed[..prefix.len()], prefix);
        assert_eq!(failed.len(), finalized + 17);
        let expected = crate::encode_frames(
            &image,
            &[
                Frame {
                    image: &image,
                    options: first,
                },
                Frame::new(&image),
            ],
            &options,
        )
        .unwrap();
        assert_eq!(
            bare_codestream(&failed[..finalized]),
            bare_codestream(&expected)
        );
        assert!(encoder.is_poisoned());
        assert!(encoder.writer.seek(finalized - 1).is_err());
        let mut retry = source(&image);
        assert!(
            encoder
                .write_frame(&mut retry, &FrameOptions::default(), &options, true)
                .is_err()
        );
        assert_eq!(retry.calls.get(), 0);
        assert!(encoder.write_metadata(*b"xml ", b"<retry/>").is_err());
        assert!(encoder.finish().is_err());
        assert_eq!(*shared.borrow().get_ref(), failed);
    }
}

#[test]
fn incremental_index_payload_limits_track_varint_boundaries() {
    let image = gray(1, 1);
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation::default());
    let image = image.with_metadata(metadata, vec![]).unwrap();
    let geometry = (&image).into();
    let options = options(1);
    let frame = FrameOptions {
        duration: 1,
        ..Default::default()
    };
    let make_index = |max_payload_bytes| {
        IndexState::new(
            FrameIndexOptions {
                max_entries: 128,
                max_payload_bytes,
            },
            geometry,
        )
        .unwrap()
    };

    // The 128th selected entry adds three fields AND widens NF from one byte to two.
    for maximum in [393, 394] {
        let mut index = make_index(maximum);
        for offset in 1..=127 {
            let update = index
                .prepare(geometry, geometry, &frame, &options, false, true, offset)
                .unwrap();
            index.commit(update);
        }
        let before = index.payload().unwrap();
        assert_eq!(before.len(), 390);
        assert_eq!(before[0], 127);
        let update = index.prepare(geometry, geometry, &frame, &options, true, true, 128);
        if maximum == 393 {
            assert!(matches!(
                update,
                Err(Error::Limit("frame index payload bytes"))
            ));
            assert_eq!(index.payload().unwrap(), before);
        } else {
            index.commit(update.unwrap());
            let payload = index.payload().unwrap();
            assert_eq!(payload.len(), 394);
            assert_eq!(&payload[..2], &[0x80, 1]);
        }
    }

    for (delta, maximum, accepted) in [(127, 15, true), (128, 15, false), (128, 16, true)] {
        let mut index = make_index(maximum);
        let first = index
            .prepare(geometry, geometry, &frame, &options, false, true, 1)
            .unwrap();
        index.commit(first);
        let before = index.payload().unwrap();
        let update = index.prepare(geometry, geometry, &frame, &options, true, true, 1 + delta);
        assert_eq!(update.is_ok(), accepted);
        if let Ok(update) = update {
            index.commit(update);
            let payload = index.payload().unwrap();
            assert_eq!(payload.len(), maximum);
            assert_eq!(
                &payload[12..],
                if delta == 127 {
                    &[127, 1, 1][..]
                } else {
                    &[0x80, 1, 1, 1][..]
                }
            );
        } else {
            assert_eq!(index.payload().unwrap(), before);
        }
    }

    // Unselected frames can widen an existing entry's Ti or Fi without adding NF.
    for displays in [false, true] {
        let before_limit = if displays { 13 } else { 12 };
        for maximum in [before_limit, before_limit + 1] {
            let mut index = make_index(maximum);
            let first = FrameOptions {
                duration: if displays { 128 } else { 127 },
                ..Default::default()
            };
            let update = index
                .prepare(geometry, geometry, &first, &options, false, true, 1)
                .unwrap();
            index.commit(update);
            if displays {
                for offset in 2..=127 {
                    let update = index
                        .prepare(geometry, geometry, &frame, &options, false, false, offset)
                        .unwrap();
                    index.commit(update);
                }
            }
            let before = index.payload().unwrap();
            assert_eq!(before.len(), before_limit);
            let update = index.prepare(geometry, geometry, &frame, &options, true, false, 128);
            if maximum == before_limit {
                assert!(matches!(
                    update,
                    Err(Error::Limit("frame index payload bytes"))
                ));
                assert_eq!(index.payload().unwrap(), before);
            } else {
                index.commit(update.unwrap());
                let payload = index.payload().unwrap();
                assert_eq!(payload.len(), maximum);
                assert_eq!(
                    &payload[10..],
                    if displays {
                        &[0xff, 1, 0x80, 1][..]
                    } else {
                        &[0x80, 1, 2][..]
                    }
                );
            }
        }
    }
}

fn custom_float_words(bits: u8, exponent_bits: u8) -> Vec<u32> {
    let mantissa = bits - exponent_bits - 1;
    let sign = 1u32 << (bits - 1);
    let infinity = ((1u32 << exponent_bits) - 1) << mantissa;
    let one = ((1u32 << (exponent_bits - 1)) - 1) << mantissa;
    vec![
        0,
        sign,
        1,
        (1 << mantissa) - 1,
        1 << mantissa,
        one,
        sign | one,
        infinity,
        sign | infinity,
        infinity | 1,
        infinity | (1 << (mantissa - 1)),
        sign | infinity | 3,
    ]
}

fn custom_float_pixels(words: &[u32], depth: BitDepth) -> PixelData {
    let input: Vec<_> = words.iter().map(|&word| word as i32).collect();
    let mut values = vec![0.0; words.len()];
    crate::modular::sample_convert::int_to_float(
        &input,
        &mut values,
        crate::modular::sample_convert::SampleFormat::Float {
            bits: depth.bits_per_sample,
            exponent_bits: depth.exponent_bits_per_sample,
        },
        1.0,
    )
    .unwrap();
    PixelData::F32(values)
}

#[test]
fn buffered_custom_float_and_mixed_typed_extras_preserve_declared_words() {
    for (bits, exponent_bits) in [(16, 6), (24, 7)] {
        let depth = BitDepth::floating(bits, exponent_bits).unwrap();
        let main_words: Vec<_> = custom_float_words(bits, exponent_bits)
            .into_iter()
            .cycle()
            .take(17 * 9)
            .collect();
        let image = Image::new(
            17,
            9,
            PixelLayout::Gray,
            custom_float_pixels(&main_words, depth),
        )
        .unwrap();
        let mut metadata = image.metadata().clone();
        metadata.bit_depth = depth;
        let specs = [
            (BitDepth::new(7).unwrap(), 0, vec![0, 127, 1, 91]),
            (BitDepth::new(12).unwrap(), 1, vec![0, 4095, 2049, 1]),
            (
                BitDepth::new(31).unwrap(),
                3,
                vec![0, 0x7fff_ffff, 0x0100_0001, 0x4000_0001],
            ),
            (
                BitDepth::floating(16, 5).unwrap(),
                2,
                vec![0, 0x8000, 0x7c01, 0xfc00, 0xfe55, 1],
            ),
            (
                BitDepth::floating(32, 8).unwrap(),
                1,
                vec![0, 0x8000_0000, 0x7f80_0001, 0xffc0_0055, 1],
            ),
            (
                BitDepth::floating(16, 6).unwrap(),
                3,
                custom_float_words(16, 6),
            ),
        ];
        let mut expected = vec![main_words];
        let mut extras = Vec::new();
        for (index, (depth, shift, pattern)) in specs.into_iter().enumerate() {
            let count = 17usize.div_ceil(1 << shift) * 9usize.div_ceil(1 << shift);
            let words: Vec<_> = pattern.into_iter().cycle().take(count).collect();
            let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
            extra.bit_depth = depth;
            extra.dim_shift = shift;
            metadata.extra_channels.push(extra);
            extras.push(match index {
                0 => PixelData::U8(words.iter().map(|&word| word as u8).collect()),
                1 => PixelData::U16(words.iter().map(|&word| word as u16).collect()),
                2 => PixelData::U32(words.clone()),
                3 => PixelData::F16(words.iter().map(|&word| word as u16).collect()),
                4 => PixelData::F32(words.iter().copied().map(f32::from_bits).collect()),
                _ => custom_float_pixels(&words, depth),
            });
            expected.push(words);
        }
        let image = image.with_metadata(metadata, extras).unwrap();
        for (channel, words) in expected.iter().enumerate() {
            for (sample, &word) in words.iter().enumerate() {
                assert_eq!(
                    image.modular_sample(channel, sample) as u32,
                    word,
                    "fixture format={bits}/{exponent_bits}, channel={channel}, sample={sample}"
                );
            }
        }
        assert_buffered_words(&image);
        let options = options(1);
        let expected = crate::encode_frames(&image, &[Frame::new(&image)], &options).unwrap();
        let mut encoder = FrameEncoder::new(
            ImageDescriptor::from_image(&image).unwrap(),
            config(1),
            Cursor::new(Vec::new()),
        )
        .unwrap();
        let progress = encoder
            .write_frame(
                &mut source(&image),
                &FrameOptions::default(),
                &options,
                true,
            )
            .unwrap();
        assert_eq!(progress.mode, FrameProgressMode::CurrentFrameBuffered);
        let (output, _) = encoder.finish().unwrap();
        assert_eq!(
            bare_codestream(output.get_ref()),
            bare_codestream(&expected)
        );
    }
}
