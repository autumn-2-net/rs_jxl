use super::*;
use crate::color::ColorEncoding;
use crate::metadata::ExtraChannelInfo;
use crate::modular::{
    encode::FixedTree, layout, orchestration, palette::PaletteOptions, predictor::Predictor,
};
use crate::{EncodeOptions, execution};
use std::cell::Cell;
use std::rc::Rc;

fn descriptor(width: u32, height: u32) -> ImageDescriptor {
    ImageDescriptor::new(
        width,
        height,
        PixelLayout::Gray,
        ImageMetadata::new(8, ColorEncoding::srgb(ColorSpace::Gray)).unwrap(),
    )
    .unwrap()
}

fn options() -> EncodeOptions {
    EncodeOptions {
        group_size_shift: 0,
        modular: crate::ModularOptions {
            predictor: Predictor::Gradient,
            rct: None,
            fixed_tree: Some(FixedTree::GradientLeaf),
            lz77: false,
            palette_policy: Some(PaletteOptions {
                groupwise: true,
                palette_colors: 0,
                channel_colors_percent: 0.0,
                channel_colors_pre_transform_percent: 0.0,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn descriptor_validates_metadata_without_pixel_storage() {
    let large = descriptor(1 << 30, 1 << 10);
    assert_eq!(large.channel_dimensions(0).unwrap(), (1 << 30, 1 << 10));
    for (width, height) in [(0, 1), (1, 0), ((1 << 30) + 1, 1), (1 << 30, (1 << 10) + 1)] {
        assert!(
            ImageDescriptor::new(width, height, PixelLayout::Gray, large.metadata().clone())
                .is_err()
        );
    }
    assert!(ImageDescriptor::new(1, 1, PixelLayout::Rgb, large.metadata().clone()).is_err());
    assert!(ImageDescriptor::new(1, 1, PixelLayout::GrayAlpha, large.metadata().clone()).is_err());
    assert!(large.channel_depth(1).is_err());
    let mut metadata = large.metadata().clone();
    let mut depth = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
    depth.dim_shift = 3;
    metadata.extra_channels.push(depth);
    let shifted = ImageDescriptor::new(17, 9, PixelLayout::Gray, metadata).unwrap();
    assert_eq!(shifted.channel_dimensions(1).unwrap(), (3, 2));
    assert!(
        shifted
            .validate_request(RegionRequest {
                plane: RegionPlane::Extra(0),
                rect: RegionRect {
                    x: 2,
                    y: 1,
                    width: 1,
                    height: 1
                },
            })
            .is_ok()
    );
    assert!(
        shifted
            .validate_request(RegionRequest {
                plane: RegionPlane::Extra(0),
                rect: RegionRect {
                    x: 3,
                    y: 1,
                    width: 1,
                    height: 1
                },
            })
            .is_err()
    );
}

#[test]
fn leases_release_once_on_success_error_and_unwind() {
    let released = Cell::new(0);
    let data = [7u8];
    let lease = || {
        RegionLease::with_release(
            RegionView {
                samples: SampleSlice::U8(&data),
                row_stride: 1,
            },
            || released.set(released.get() + 1),
        )
    };
    drop(lease());
    fn fail(_lease: RegionLease<'_>) -> Result<()> {
        Err(Error::Limit("test failure"))
    }
    assert!(fail(lease()).is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = lease();
            panic!("test unwind");
        }))
        .is_err()
    );
    assert_eq!(released.get(), 3);
}

fn assert_loaded(image: &Image) {
    let descriptor = ImageDescriptor::from_image(image).unwrap();
    let mut source = ImageRegionSource::new(image).unwrap();
    let area = layout::Rect {
        x: 2,
        y: 1,
        width: 3,
        height: 2,
    };
    let channels = load::read_area(&descriptor, &mut source, area).unwrap();
    for (channel, plane) in channels.iter().enumerate() {
        let shift = image.channel_shift(channel);
        let (full_width, _) = image.channel_dimensions(channel);
        for y in 0..plane.height {
            for x in 0..plane.width {
                let expected = image.modular_sample(
                    channel,
                    ((area.y >> shift) + y) * full_width + (area.x >> shift) + x,
                );
                assert_eq!(plane.samples[y * plane.width + x], expected);
            }
        }
    }
}

#[test]
fn borrowed_image_regions_preserve_native_words_mixed_depths_and_subsampling() {
    for pixels in [
        PixelData::U8((0..15).map(|i| (i * 13) as u8).collect()),
        PixelData::U16((0..15).map(|i| (i * 4097) as u16).collect()),
        PixelData::U32((0..15).map(|i| (i * 137_000_003) as u32).collect()),
        PixelData::F16([0x8000, 0x7e01, 0x3c00].repeat(5)),
        PixelData::F32([-0.0, f32::from_bits(0x7fc12345), 1.0].repeat(5)),
    ] {
        let image = Image::new(5, 3, PixelLayout::Gray, pixels).unwrap();
        assert_loaded(&image);
    }
    let image = Image::new(
        5,
        3,
        PixelLayout::Rgba,
        PixelData::U8((0..15).flat_map(|i| [i as u8, 97, 211, 0]).collect()),
    )
    .unwrap();
    let mut metadata = image.metadata().clone();
    let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
    extra.dim_shift = 1;
    metadata.extra_channels.push(extra);
    let image = image
        .with_metadata(
            metadata,
            vec![PixelData::U16(vec![1, 19, 257, 4097, 65535, 0])],
        )
        .unwrap();
    assert_loaded(&image);
    let mut adapter = ImageRegionSource::new(&image).unwrap();
    let lease = adapter
        .acquire(RegionRequest {
            plane: RegionPlane::Color,
            rect: RegionRect {
                x: 2,
                y: 1,
                width: 1,
                height: 1,
            },
        })
        .unwrap();
    let view = lease.view();
    assert_eq!(view.row_stride, 20);
    let SampleSlice::U8(samples) = view.samples else {
        panic!("native storage changed")
    };
    let PixelData::U8(original) = image.pixels() else {
        unreachable!()
    };
    assert_eq!(samples.as_ptr(), original[28..].as_ptr());
    drop(lease);
    assert!(
        adapter
            .acquire(RegionRequest {
                plane: RegionPlane::Extra(0),
                rect: RegionRect {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1
                },
            })
            .is_err()
    );
    let custom = Image::new(5, 3, PixelLayout::Gray, PixelData::F32(vec![0.5; 15])).unwrap();
    let mut metadata = custom.metadata().clone();
    metadata.bit_depth = BitDepth::floating(16, 5).unwrap();
    assert_loaded(&custom.with_metadata(metadata, vec![]).unwrap());
}

// A genuinely non-Send provider with no full image backing buffer.
struct Generated {
    descriptor: ImageDescriptor,
    scratch: Vec<u8>,
    requests: Vec<RegionRequest>,
    live: Rc<Cell<usize>>,
    caller: std::thread::ThreadId,
    short: bool,
}

impl Generated {
    fn new(width: u32, height: u32) -> Self {
        Self {
            descriptor: descriptor(width, height),
            scratch: vec![],
            requests: vec![],
            live: Rc::new(Cell::new(0)),
            caller: std::thread::current().id(),
            short: false,
        }
    }
}

impl RegionSource for Generated {
    fn descriptor(&self) -> &ImageDescriptor {
        &self.descriptor
    }

    fn acquire(&mut self, request: RegionRequest) -> Result<RegionLease<'_>> {
        assert_eq!(std::thread::current().id(), self.caller);
        assert_eq!(self.live.get(), 0);
        assert!(request.rect.width <= 2048 && request.rect.height <= 2048);
        self.descriptor.validate_request(request)?;
        self.requests.push(request);
        let stride = request.rect.width + 3;
        self.scratch.resize(stride * request.rect.height, 0);
        for y in 0..request.rect.height {
            for x in 0..request.rect.width {
                self.scratch[y * stride + x] =
                    ((request.rect.x + x + request.rect.y + y) % 251) as u8;
            }
        }
        if self.short {
            self.scratch.clear();
        }
        self.live.set(1);
        let live = Rc::clone(&self.live);
        Ok(RegionLease::with_release(
            RegionView {
                samples: SampleSlice::U8(&self.scratch),
                row_stride: stride,
            },
            move || live.set(live.get() - 1),
        ))
    }
}

#[test]
fn plan_exposes_absolute_topology_and_rejects_ineligible_options() {
    for shift in 0..=3 {
        let mut options = options();
        options.group_size_shift = shift;
        let plan = FrameAreaPlan::new(descriptor((128 << shift) * 9 + 1, 1), &options).unwrap();
        assert_eq!(
            (
                plan.num_groups(),
                plan.num_dc_groups(),
                plan.global_section_count()
            ),
            (10, 2, 2)
        );
        assert_eq!(plan.num_streams(), 34);
        let mut canonical: Vec<_> = plan.sections().iter().map(|s| s.canonical_index).collect();
        canonical.sort_unstable();
        assert_eq!(canonical, (0..plan.section_count()).collect::<Vec<_>>());
        assert_eq!(plan.sections()[0].kind, SectionKind::DcGlobal);
        assert_eq!(plan.sections().last().unwrap().kind, SectionKind::AcGlobal);
        for section in plan.sections() {
            match section.kind {
                SectionKind::DcGroup(group) => assert_eq!(section.stream_id, Some(3 + group)),
                SectionKind::AcGroup(group) => assert_eq!(section.stream_id, Some(24 + group)),
                _ => {}
            }
        }
    }
    for case in 0..7 {
        let mut options = options();
        match case {
            0 => options.modular.palette_policy.as_mut().unwrap().groupwise = false,
            1 => options.responsive = true,
            2 => options.passes = layout::Passes::progressive(),
            3 => options.source_patches = Some(Default::default()),
            4 => options.modular.squeeze.push(true),
            5 => options.max_pixels = 1,
            6 => options.modular.sample_limit = 0,
            _ => unreachable!(),
        }
        assert!(FrameAreaPlan::new(descriptor(1025, 1), &options).is_err());
    }
    assert!(FrameAreaPlan::new(descriptor(1024, 1), &options()).is_err());
}

#[test]
fn non_send_callbacks_stay_on_caller_and_errors_stop_before_next_area() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for use_pool in [false, true] {
        for fail in [false, true] {
            let mut source = Generated::new(1025, 1);
            let mut options = options();
            options.threads = 2;
            let plan = FrameAreaPlan::new(source.descriptor().clone(), &options).unwrap();
            let live = Rc::clone(&source.live);
            let caller = source.caller;
            let mut seen = Vec::new();
            let result = plan.emit(&mut source, use_pool.then_some(&pool), |section, bytes| {
                assert_eq!(std::thread::current().id(), caller);
                assert_eq!(live.get(), 0);
                if fail {
                    return Err(Error::Limit("sink stop"));
                }
                seen.push((section, bytes.len()));
                Ok(())
            });
            assert_eq!(source.live.get(), 0);
            if fail {
                assert!(matches!(result, Err(Error::Limit("sink stop"))));
                assert_eq!(source.requests.len(), 1);
            } else {
                assert_eq!(
                    result.unwrap(),
                    seen.iter().map(|(_, size)| size).sum::<usize>()
                );
                assert_eq!(
                    seen.iter().map(|(section, _)| *section).collect::<Vec<_>>(),
                    plan.sections()
                );
                assert_eq!(source.requests.len(), 2);
                assert_eq!(seen.last().unwrap().1, 0);
            }
        }
    }
    let mut source = Generated::new(1025, 1);
    source.short = true;
    let plan = FrameAreaPlan::new(source.descriptor().clone(), &options()).unwrap();
    assert!(
        plan.emit(&mut source, None, |_, _| panic!(
            "invalid input emitted bytes"
        ))
        .is_err()
    );
    assert_eq!(source.live.get(), 0);
}

#[test]
fn region_sections_match_existing_area_encoder_without_rescaling_sampling() {
    use crate::modular::sample::{SamplingPolicy, SourcePolicy};
    let image = Image::new(
        1025,
        1,
        PixelLayout::Gray,
        PixelData::U8((0..1025).map(|i| (i % 251) as u8).collect()),
    )
    .unwrap();
    for fraction in [None, Some(0.0), Some(0.125)] {
        let mut options = options();
        if let Some(fraction) = fraction {
            options.modular.fixed_tree = None;
            options.modular.max_tree_leaves = crate::modular::ma_train::MAX_LEAVES;
            options.modular.sampling = Some(
                SamplingPolicy::from_source(&SourcePolicy {
                    speed_tier: 1,
                    predictor: Predictor::Gradient,
                    learning_fraction: fraction,
                    num_streams: 33,
                    max_channel_size: 128,
                    is_gray: true,
                    ..Default::default()
                })
                .unwrap(),
            );
        }
        let expected = execution::serial(|| {
            orchestration::encode_layout_trial(
                &image,
                &options.modular,
                0,
                &options.passes,
                false,
                false,
                options.max_output_bytes,
                None,
                None,
            )
        })
        .unwrap()
        .into_complete()
        .unwrap();
        let mut source = ImageRegionSource::new(&image).unwrap();
        let plan = FrameAreaPlan::new(source.descriptor().clone(), &options).unwrap();
        let mut actual = vec![Vec::new(); plan.section_count()];
        plan.emit(&mut source, None, |section, bytes| {
            actual[section.canonical_index] = bytes.to_vec();
            Ok(())
        })
        .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn wide_area_input_is_tiled_and_private_working_images_are_not_reinterpreted() {
    let mut source = Generated::new(8193, 1);
    let channels = load::read_area(
        &source.descriptor().clone(),
        &mut source,
        layout::Rect {
            x: 0,
            y: 0,
            width: 8192,
            height: 1,
        },
    )
    .unwrap();
    assert_eq!(source.requests.len(), 4);
    assert_eq!(source.live.get(), 0);
    assert_eq!(channels[0].samples[8191], 8191 % 251);
    let image = Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![0])).unwrap();
    let working = image.with_modular_planes(1, 1, vec![vec![-3]]).unwrap();
    assert!(ImageDescriptor::from_image(&working).is_ok());
    assert!(ImageRegionSource::new(&working).is_err());
}
