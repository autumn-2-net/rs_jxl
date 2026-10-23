//! Independent push-sequence acceptance. Run ignored tests in release mode with
//! RS_JXL_DJXL listing official 0.12 and 0.13 executables (OS path-list syntax).
//! Expected samples are composed here, never decoded from another JXL encoder.
//! Child processes isolate profiling; successful runs print summaries only.

use std::cell::RefCell;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::frame::{BlendInfo, FrameOptions};
use rs_jxl::metadata::{Animation, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::modular::{encode::FixedTree, palette::PaletteOptions, predictor::Predictor};
use rs_jxl::region_input::{
    FrameAreaPlan, ImageDescriptor, RegionLease, RegionPlane, RegionRequest, RegionSource,
    RegionView, SampleSlice,
};
use rs_jxl::streaming::{
    FrameBuffering, FrameEncoder, FrameEncoderOptions, FrameIndexOptions, FrameProgressMode,
};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout};

const CHILD: &str = "RS_JXL_SEQUENCE_TEST_CHILD";
const CHILD_OUTPUT: &str = "RS_JXL_SEQUENCE_TEST_OUTPUT";
const OUTPUT_LIMIT: usize = 32 << 20;
const LOG_LIMIT: u64 = 16 << 20;
const SENTINEL: &[u8] = b"existing caller data\0";
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct Submission {
    image: Image,
    frame: FrameOptions,
    mode: FrameProgressMode,
}

struct Fixture {
    canvas: ImageDescriptor,
    submissions: Vec<Submission>,
    expected: Vec<Image>,
    buffering: FrameBuffering,
    indexed_frames: Vec<usize>,
}

fn options(threads: usize) -> EncodeOptions {
    EncodeOptions {
        threads,
        group_size_shift: 0,
        max_pixels: 1 << 20,
        max_output_bytes: OUTPUT_LIMIT,
        modular: ModularOptions {
            predictor: Predictor::Gradient,
            fixed_tree: Some(FixedTree::GradientLeaf),
            rct: None,
            lz77: false,
            palette_policy: Some(PaletteOptions {
                groupwise: true,
                palette_colors: 0,
                channel_colors_pre_transform_percent: 0.0,
                channel_colors_percent: 0.0,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rgba(width: u32, height: u32, seed: u32) -> Image {
    let mut samples = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            samples.extend_from_slice(&[
                (17 + x * 13 + y * 3 + seed * 29) as u8,
                (223 + x * 7 + y * 19 + seed * 11) as u8,
                (91 + x * 5 + y * 23 + seed * 17) as u8,
                if (x + y * width).is_multiple_of(17) {
                    0
                } else {
                    (x * 11 + y * 7 + seed) as u8
                },
            ]);
        }
    }
    samples[..4].copy_from_slice(&[17, 223, 91, 0]);
    Image::new(width, height, PixelLayout::Rgba, PixelData::U8(samples)).unwrap()
}

fn canvas(image: &Image) -> ImageDescriptor {
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation {
        ticks_per_second_numerator: 24,
        ..Default::default()
    });
    ImageDescriptor::new(image.width(), image.height(), image.layout(), metadata).unwrap()
}

fn replace(base: &Image, patch: &Image, origin: (usize, usize)) -> Image {
    assert_eq!(base.layout(), patch.layout());
    assert!(base.extra_planes().is_empty() && patch.extra_planes().is_empty());
    assert!(origin.0 + patch.width() as usize <= base.width() as usize);
    assert!(origin.1 + patch.height() as usize <= base.height() as usize);
    let components = base.layout().channels();
    let copy = |out: &mut PixelData| {
        for row in 0..patch.height() as usize {
            let start = ((origin.1 + row) * base.width() as usize + origin.0) * components;
            let length = patch.width() as usize * components;
            let source = row * length;
            match (&mut *out, patch.pixels()) {
                (PixelData::U8(a), PixelData::U8(b)) => {
                    a[start..start + length].copy_from_slice(&b[source..source + length])
                }
                (PixelData::U16(a), PixelData::U16(b)) | (PixelData::F16(a), PixelData::F16(b)) => {
                    a[start..start + length].copy_from_slice(&b[source..source + length])
                }
                _ => panic!("unsupported fixture storage"),
            }
        }
    };
    let mut pixels = base.pixels().clone();
    copy(&mut pixels);
    Image::new(base.width(), base.height(), base.layout(), pixels)
        .unwrap()
        .with_metadata(base.metadata().clone(), vec![])
        .unwrap()
}

fn cropped(origin: (i32, i32), source: u8, save: u8, duration: u32) -> FrameOptions {
    FrameOptions {
        origin,
        duration,
        save_as_reference: save,
        blend: BlendInfo {
            source,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mixed_fixture() -> Fixture {
    let a = rgba(513, 513, 1);
    let b = rgba(513, 513, 2);
    let p1 = rgba(13, 17, 3);
    let p2 = rgba(7, 5, 4);
    let p3 = rgba(3, 2, 5);
    // Frame 2 must restore reference 1, not the more recent full-frame layer B.
    let first = replace(&a, &p1, (11, 9));
    let last = replace(&replace(&first, &p2, (31, 27)), &p3, (101, 93));
    Fixture {
        canvas: canvas(&a),
        submissions: vec![
            Submission {
                image: a,
                frame: FrameOptions {
                    reference_only: true,
                    save_as_reference: 1,
                    ..Default::default()
                },
                mode: FrameProgressMode::FrameAreas,
            },
            Submission {
                image: b,
                frame: FrameOptions {
                    save_as_reference: 2,
                    ..Default::default()
                },
                mode: FrameProgressMode::FrameAreas,
            },
            Submission {
                image: p1,
                frame: cropped((11, 9), 1, 3, 5),
                mode: FrameProgressMode::CurrentFrameBuffered,
            },
            Submission {
                image: p2,
                frame: cropped((31, 27), 3, 2, 0),
                mode: FrameProgressMode::CurrentFrameBuffered,
            },
            Submission {
                image: p3,
                frame: cropped((101, 93), 2, 0, 0),
                mode: FrameProgressMode::CurrentFrameBuffered,
            },
        ],
        expected: vec![first, last],
        // Too small to accidentally buffer either full RGBA frame (i64 staging).
        buffering: FrameBuffering::CurrentFrame {
            max_sample_bytes: 1 << 20,
        },
        indexed_frames: vec![],
    }
}

fn extras_fixture() -> Fixture {
    let expected: Vec<_> = [0u16, 32768, 65535]
        .into_iter()
        .enumerate()
        .map(|(frame, constant)| {
            let image = rgba(513, 513, frame as u32 + 7);
            let mut metadata = image.metadata().clone();
            let full = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
            let mut half = full.clone();
            half.dim_shift = 1;
            metadata.extra_channels.extend([full, half]);
            let mut depth: Vec<_> = (0..513u32 * 513)
                .map(|i| (i * 977 + (i / 513) * 7919 + frame as u32 * 113) as u16)
                .collect();
            depth[..3].copy_from_slice(&[0, 32768, 65535]);
            image
                .with_metadata(
                    metadata,
                    vec![
                        PixelData::U16(depth),
                        // Native ceil(513/2)^2 storage. Constant planes have exact
                        // expected upsampled values without an interpolation oracle.
                        PixelData::U16(vec![constant; 257 * 257]),
                    ],
                )
                .unwrap()
        })
        .collect();
    Fixture {
        canvas: canvas(&expected[0]),
        submissions: expected
            .iter()
            .enumerate()
            .map(|(index, image)| Submission {
                image: image.clone(),
                frame: FrameOptions {
                    duration: [3, 5, 7][index],
                    ..Default::default()
                },
                mode: FrameProgressMode::FrameAreas,
            })
            .collect(),
        expected,
        buffering: FrameBuffering::StrictAreas,
        indexed_frames: vec![],
    }
}

fn rgba16(width: u32, height: u32, seed: u32) -> Image {
    let mut samples = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            samples.extend_from_slice(&[
                (x * 997 + y * 73 + seed * 8191) as u16,
                (x * 1237 + y * 113 + seed * 911) as u16,
                (x * 31 + y * 9973 + seed * 211) as u16,
                if (x + y * width).is_multiple_of(17) {
                    0
                } else {
                    (x * 137 + y * 101 + seed * 997) as u16
                },
            ]);
        }
    }
    samples[..4].copy_from_slice(&[1, 32768, 65535, 0]);
    Image::new(width, height, PixelLayout::Rgba, PixelData::U16(samples)).unwrap()
}

fn indexed_fixture() -> Fixture {
    let a = rgba16(67, 53, 1);
    let p1 = rgba16(13, 17, 2);
    let p2 = rgba16(3, 2, 3);
    let last = rgba16(67, 53, 4);
    let composed = replace(&replace(&a, &p1, (11, 9)), &p2, (31, 27));
    Fixture {
        canvas: canvas(&a),
        expected: vec![a.clone(), composed, last.clone()],
        submissions: vec![
            Submission {
                image: a,
                frame: FrameOptions {
                    duration: 11,
                    save_as_reference: 1,
                    ..Default::default()
                },
                mode: FrameProgressMode::CurrentFrameBuffered,
            },
            Submission {
                image: p1,
                frame: cropped((11, 9), 1, 2, 0),
                mode: FrameProgressMode::CurrentFrameBuffered,
            },
            Submission {
                image: p2,
                frame: cropped((31, 27), 2, 0, 7),
                mode: FrameProgressMode::CurrentFrameBuffered,
            },
            Submission {
                image: last,
                frame: FrameOptions::default(),
                mode: FrameProgressMode::CurrentFrameBuffered,
            },
        ],
        buffering: FrameBuffering::CurrentFrame {
            max_sample_bytes: 1 << 20,
        },
        indexed_frames: vec![0, 3],
    }
}

fn half_image(width: u32, height: u32, seed: usize) -> Image {
    const WORDS: [u16; 14] = [
        0, 0x8000, 1, 0x03ff, 0x0400, 0x3c00, 0x7c00, 0xfc00, 0x7e01, 0xfe55, 0x7c01, 0xfc01,
        0x7fff, 0xffff,
    ];
    let samples = (0..width as usize * height as usize)
        .map(|i| WORDS[(i + seed) % WORDS.len()])
        .collect();
    Image::new(width, height, PixelLayout::Gray, PixelData::F16(samples)).unwrap()
}

fn half_fixture() -> Fixture {
    let a = half_image(37, 19, 0);
    let p1 = half_image(17, 3, 5);
    let p2 = half_image(3, 2, 9);
    let last = replace(&replace(&a, &p1, (5, 7)), &p2, (25, 13));
    Fixture {
        canvas: canvas(&a),
        expected: vec![a.clone(), last],
        submissions: vec![
            Submission {
                image: a,
                frame: FrameOptions {
                    duration: 2,
                    save_as_reference: 1,
                    ..Default::default()
                },
                mode: FrameProgressMode::CurrentFrameBuffered,
            },
            Submission {
                image: p1,
                frame: cropped((5, 7), 1, 2, 0),
                mode: FrameProgressMode::CurrentFrameBuffered,
            },
            Submission {
                image: p2,
                frame: cropped((25, 13), 2, 0, 0),
                mode: FrameProgressMode::CurrentFrameBuffered,
            },
        ],
        buffering: FrameBuffering::CurrentFrame {
            max_sample_bytes: 1 << 20,
        },
        indexed_frames: vec![],
    }
}

#[derive(Default)]
struct SourceStats {
    requests: Vec<RegionRequest>,
    releases: usize,
    live: usize,
}

struct Source<'a> {
    image: &'a Image,
    descriptor: ImageDescriptor,
    stats: Rc<RefCell<SourceStats>>,
    caller: ThreadId,
}

impl RegionSource for Source<'_> {
    fn descriptor(&self) -> &ImageDescriptor {
        assert_eq!(thread::current().id(), self.caller);
        &self.descriptor
    }

    fn acquire(&mut self, request: RegionRequest) -> rs_jxl::Result<RegionLease<'_>> {
        assert_eq!(thread::current().id(), self.caller);
        self.descriptor.validate_request(request)?;
        let (pixels, width, components) = match request.plane {
            RegionPlane::Color => (
                self.image.pixels(),
                self.image.width() as usize,
                self.image.layout().channels(),
            ),
            RegionPlane::Extra(extra) => (
                &self.image.extra_planes()[extra - usize::from(self.image.layout().has_alpha())],
                self.descriptor
                    .channel_dimensions(self.image.color_channels() + extra)?
                    .0,
                1,
            ),
        };
        let stride = width * components;
        let offset = request.rect.y * stride + request.rect.x * components;
        let samples = match pixels {
            PixelData::U8(v) => SampleSlice::U8(&v[offset..]),
            PixelData::U16(v) => SampleSlice::U16(&v[offset..]),
            PixelData::U32(v) => SampleSlice::U32(&v[offset..]),
            PixelData::F16(v) => SampleSlice::F16(&v[offset..]),
            PixelData::F32(v) => SampleSlice::F32(&v[offset..]),
        };
        {
            let mut stats = self.stats.borrow_mut();
            assert_eq!(stats.live, 0, "only one live region lease");
            stats.live = 1;
            stats.requests.push(request);
        }
        let stats = self.stats.clone();
        let caller = self.caller;
        Ok(RegionLease::with_release(
            RegionView {
                samples,
                row_stride: stride,
            },
            move || {
                assert_eq!(thread::current().id(), caller);
                let mut stats = stats.borrow_mut();
                assert_eq!(stats.live, 1);
                stats.live = 0;
                stats.releases += 1;
            },
        ))
    }
}

struct SinkState {
    cursor: Cursor<Vec<u8>>,
    floor: usize,
}

struct Sink(Rc<RefCell<SinkState>>, ThreadId);

impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        assert_eq!(thread::current().id(), self.1);
        let mut state = self.0.borrow_mut();
        let position = usize::try_from(state.cursor.position()).unwrap();
        assert!(position >= state.floor, "write into finalized prefix");
        assert!(position.checked_add(bytes.len()).unwrap() <= SENTINEL.len() + OUTPUT_LIMIT);
        state.cursor.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        panic!("sequence API must not flush caller output")
    }
}

impl Seek for Sink {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        assert_eq!(thread::current().id(), self.1);
        let mut state = self.0.borrow_mut();
        let position = state.cursor.seek(position)?;
        assert!(position >= state.floor as u64, "seek into finalized prefix");
        Ok(position)
    }
}

fn freeze_prefix(shared: &RefCell<SinkState>, previous: &mut Vec<u8>, finalized: usize) {
    let mut state = shared.borrow_mut();
    let bytes = state.cursor.get_ref();
    assert_eq!(bytes.len(), SENTINEL.len() + finalized);
    assert_eq!(&bytes[..SENTINEL.len()], SENTINEL);
    assert_eq!(
        &bytes[SENTINEL.len()..SENTINEL.len() + previous.len()],
        previous
    );
    *previous = bytes[SENTINEL.len()..].to_vec();
    state.floor = SENTINEL.len() + finalized;
}

fn encode_fixture(fixture: Fixture, threads: usize) -> Vec<u8> {
    let count = fixture.submissions.len();
    let mut cursor = Cursor::new(SENTINEL.to_vec());
    cursor.set_position(SENTINEL.len() as u64);
    let shared = Rc::new(RefCell::new(SinkState {
        cursor,
        floor: SENTINEL.len(),
    }));
    let mut encoder = FrameEncoder::new(
        fixture.canvas,
        FrameEncoderOptions {
            threads,
            max_output_bytes: OUTPUT_LIMIT,
            buffering: fixture.buffering,
            index: (!fixture.indexed_frames.is_empty()).then_some(FrameIndexOptions {
                max_entries: fixture.indexed_frames.len(),
                max_payload_bytes: 128,
            }),
        },
        Sink(shared.clone(), thread::current().id()),
    )
    .unwrap();
    assert_eq!(shared.borrow().cursor.get_ref().len(), SENTINEL.len());
    let mut prefix = Vec::new();
    let mut areas = 0;
    let mut requests = 0;
    for (index, submission) in fixture.submissions.into_iter().enumerate() {
        if !fixture.indexed_frames.is_empty() && index == 2 {
            let finalized = encoder.write_metadata(*b"xml ", b"<sequence/>").unwrap();
            assert_eq!(encoder.finalized_bytes(), finalized);
            assert!(finalized > prefix.len());
            freeze_prefix(&shared, &mut prefix, finalized);
        }
        let stats = Rc::new(RefCell::new(SourceStats::default()));
        let mut source = Source {
            descriptor: ImageDescriptor::from_image(&submission.image).unwrap(),
            image: &submission.image,
            stats: stats.clone(),
            caller: thread::current().id(),
        };
        let progress = if fixture.indexed_frames.contains(&index) {
            encoder.write_indexed_frame(
                &mut source,
                &submission.frame,
                &options(threads),
                index + 1 == count,
            )
        } else {
            encoder.write_frame(
                &mut source,
                &submission.frame,
                &options(threads),
                index + 1 == count,
            )
        }
        .unwrap();
        assert_eq!(progress.frame_index as usize, index);
        assert_eq!(progress.mode, submission.mode);
        assert_eq!(progress.is_last, index + 1 == count);
        assert_eq!(progress.bytes_written, progress.finalized_bytes);
        assert_eq!(encoder.finalized_bytes(), progress.finalized_bytes);
        assert!(progress.finalized_bytes > prefix.len());
        assert!(!encoder.is_poisoned());
        freeze_prefix(&shared, &mut prefix, progress.finalized_bytes);
        let stats = stats.borrow();
        assert!(!stats.requests.is_empty());
        assert_eq!(stats.live, 0);
        assert_eq!(stats.releases, stats.requests.len());
        for extra in usize::from(submission.image.layout().has_alpha())
            ..submission.image.metadata().extra_channels.len()
        {
            assert!(
                stats
                    .requests
                    .iter()
                    .any(|r| r.plane == RegionPlane::Extra(extra)),
                "extra {extra} never acquired"
            );
        }
        requests += stats.requests.len();
        areas += usize::from(progress.mode == FrameProgressMode::FrameAreas);
        // Both provider and submitted samples leave scope before the next push.
    }
    let (_, length) = encoder.finish().unwrap();
    assert_eq!(length, prefix.len());
    freeze_prefix(&shared, &mut prefix, length);
    eprintln!(
        "[sequence acceptance] frames={count} areas={areas} buffered={} requests={requests} releases={requests} finalized={length}",
        count - areas
    );
    if let Some(profile) = rs_jxl::profile::report() {
        eprintln!("{profile}");
    }
    prefix
}

#[derive(Debug, PartialEq)]
struct Npy {
    shape: [usize; 4],
    words: Vec<u32>,
}

// Fixed ASCII dictionary emitted by official enc/npy.cc, not a Python parser.
fn parse_npy(bytes: &[u8]) -> Result<Npy, String> {
    if bytes.len() < 10 || &bytes[..8] != b"\x93NUMPY\x01\0" {
        return Err("expected NPY v1".into());
    }
    let end = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let header = bytes.get(10..end).ok_or("truncated NPY header")?;
    if !header.is_ascii() || !header.ends_with(b"\n") {
        return Err("invalid ASCII NPY header".into());
    }
    let dimensions = std::str::from_utf8(header)
        .unwrap()
        .trim_end()
        .strip_prefix("{'descr': '<f4', 'fortran_order': False, 'shape': (")
        .and_then(|s| s.strip_suffix("), }"))
        .ok_or("expected official NPY dictionary")?;
    let parts: Vec<_> = dimensions.split(',').map(str::trim).collect();
    if parts.len() != 4 {
        return Err("expected four dimensions".into());
    }
    let mut shape = [0usize; 4];
    let mut count = 1usize;
    for (out, dimension) in shape.iter_mut().zip(parts) {
        if dimension.is_empty() || !dimension.bytes().all(|b| b.is_ascii_digit()) {
            return Err("invalid dimension".into());
        }
        *out = dimension.parse().map_err(|_| "dimension overflow")?;
        if *out == 0 {
            return Err("zero dimension".into());
        }
        count = count.checked_mul(*out).ok_or("shape overflow")?;
    }
    if count.checked_mul(4) != Some(bytes.len() - end) {
        return Err("NPY payload length mismatch".into());
    }
    Ok(Npy {
        shape,
        words: bytes[end..]
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect(),
    })
}

fn half_word(word: u16) -> u32 {
    let sign = u32::from(word & 0x8000) << 16;
    let exp = u32::from((word >> 10) & 31);
    let fraction = u32::from(word & 1023);
    match (exp, fraction) {
        (0, 0) => sign,
        (0, fraction) => {
            let leading = 31 - fraction.leading_zeros();
            sign | ((leading + 103) << 23) | ((fraction - (1 << leading)) << (23 - leading))
        }
        (31, _) => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exp + 112) << 23) | (fraction << 13),
    }
}

fn assert_samples(decoded: &Npy, images: &[Image], context: &str) {
    let base = &images[0];
    let channels = base.channel_count();
    let pixels = base.width() as usize * base.height() as usize;
    assert_eq!(
        decoded.shape,
        [
            images.len(),
            base.height() as usize,
            base.width() as usize,
            channels
        ],
        "{context}: displayed frames and channels"
    );
    for (frame, image) in images.iter().enumerate() {
        // This oracle deliberately uses constant subsampled planes, not an
        // approximation of the official decoder's spatial upsampling filter.
        for (extra, plane) in image.extra_planes().iter().enumerate() {
            let metadata_extra = extra + usize::from(image.layout().has_alpha());
            if image.metadata().extra_channels[metadata_extra].dim_shift != 0 {
                let PixelData::U16(values) = plane else {
                    panic!("subsampled fixture storage")
                };
                assert!(!values.is_empty() && values.iter().all(|v| *v == values[0]));
            }
        }
        for pixel in 0..pixels {
            for channel in 0..channels {
                let (data, index) = if channel < image.layout().channels() {
                    (image.pixels(), pixel * image.layout().channels() + channel)
                } else {
                    let extra = channel - image.layout().channels();
                    let metadata_extra = channel - image.color_channels();
                    let index = if image.metadata().extra_channels[metadata_extra].dim_shift == 0 {
                        pixel
                    } else {
                        0
                    };
                    (&image.extra_planes()[extra], index)
                };
                let actual = decoded.words[(frame * pixels + pixel) * channels + channel];
                let expected = match data {
                    PixelData::F16(v) => {
                        assert_eq!(
                            actual,
                            half_word(v[index]),
                            "{context}: frame={frame} pixel={pixel} channel={channel} F16 raw word"
                        );
                        continue;
                    }
                    PixelData::U8(v) => u32::from(v[index]),
                    PixelData::U16(v) => u32::from(v[index]),
                    _ => panic!("unsupported oracle storage"),
                };
                let value = f32::from_bits(actual);
                assert!(
                    value.is_finite() && (0.0..=1.0).contains(&value),
                    "{context}: invalid integer sample"
                );
                let maximum = (1u32 << image.channel_depth(channel).bits_per_sample) - 1;
                assert_eq!(
                    (f64::from(value) * f64::from(maximum)).round(),
                    f64::from(expected),
                    "{context}: frame={frame} pixel={pixel} channel={channel} exact sample"
                );
            }
        }
    }
}

struct Transport<'a> {
    prefix: &'a [u8],
    chunks: Vec<&'a [u8]>,
    index: Option<&'a [u8]>,
    xml_boxes: usize,
}

// Independent finite-box and ordered-v0 framing reader. Offsets in jxli must
// exclude both these box headers and each jxlp's four-byte sequence number.
fn parse_transport(bytes: &[u8]) -> Result<Transport<'_>, &'static str> {
    if !bytes.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n") {
        return Err("missing container signature");
    }
    let mut position = 0usize;
    let mut prefix_end = None;
    let mut chunks = Vec::new();
    let mut index = None;
    let mut complete = false;
    let mut level = None;
    let mut xml_boxes = 0;
    while position < bytes.len() {
        let rest = &bytes[position..];
        let header = rest.get(..8).ok_or("truncated box header")?;
        let size = u32::from_be_bytes(header[..4].try_into().unwrap());
        let (size, header_len) = match size {
            0 => return Err("sequence writer must use finite boxes"),
            1 => (
                usize::try_from(u64::from_be_bytes(
                    rest.get(8..16)
                        .ok_or("truncated large size")?
                        .try_into()
                        .unwrap(),
                ))
                .map_err(|_| "box size overflow")?,
                16,
            ),
            size => (size as usize, 8),
        };
        if size < header_len {
            return Err("box size smaller than header");
        }
        let payload = rest.get(header_len..size).ok_or("truncated box payload")?;
        match &header[4..8] {
            b"jxll" => {
                if level.replace(payload).is_some() {
                    return Err("duplicate level box");
                }
            }
            b"jxlp" => {
                if complete || chunks.len() >= 256 {
                    return Err("chunk after final or too many chunks");
                }
                let number = u32::from_be_bytes(
                    payload
                        .get(..4)
                        .ok_or("missing chunk number")?
                        .try_into()
                        .unwrap(),
                );
                if number & 0x7fff_ffff != chunks.len() as u32 {
                    return Err("unordered v0 chunks");
                }
                complete = number >> 31 != 0;
                prefix_end.get_or_insert(position);
                let data = &payload[4..];
                if data.is_empty() {
                    return Err("empty frame chunk");
                }
                chunks.push(data);
            }
            b"jxli" => {
                if !complete || index.replace(payload).is_some() {
                    return Err("premature or duplicate index");
                }
            }
            b"xml " => {
                if payload != b"<sequence/>" {
                    return Err("incorrect test metadata");
                }
                xml_boxes += 1;
            }
            b"JXL " | b"ftyp" => {}
            _ => return Err("unexpected sequence box"),
        }
        position = position.checked_add(size).ok_or("box offset overflow")?;
    }
    if !complete || level != Some(&[10][..]) {
        return Err("expected complete Level 10 sequence");
    }
    let prefix_end = prefix_end.ok_or("missing codestream chunks")?;
    Ok(Transport {
        prefix: &bytes[..prefix_end],
        chunks,
        index,
        xml_boxes,
    })
}

fn read_varint(bytes: &mut &[u8]) -> Result<u64, &'static str> {
    let mut value = 0;
    for shift in (0..63).step_by(7) {
        let (&byte, rest) = bytes.split_first().ok_or("truncated index varint")?;
        *bytes = rest;
        value |= u64::from(byte & 127) << shift;
        if byte & 128 == 0 {
            if shift != 0 && byte == 0 {
                return Err("noncanonical index varint");
            }
            return Ok(value);
        }
    }
    Err("index varint exceeds 63 bits")
}

#[derive(Debug, PartialEq)]
struct Index {
    ticks: [u32; 2],
    entries: Vec<[u64; 3]>,
}

fn parse_index(mut bytes: &[u8], max_entries: usize) -> Result<Index, &'static str> {
    let count = usize::try_from(read_varint(&mut bytes)?).map_err(|_| "index count overflow")?;
    if count == 0 || count > max_entries {
        return Err("index entry count limit");
    }
    let ticks = bytes.get(..8).ok_or("truncated index ticks")?;
    let ticks = [
        u32::from_be_bytes(ticks[..4].try_into().unwrap()),
        u32::from_be_bytes(ticks[4..].try_into().unwrap()),
    ];
    if ticks[1] == 0 {
        return Err("zero tick denominator");
    }
    bytes = &bytes[8..];
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let entry = [
            read_varint(&mut bytes)?,
            read_varint(&mut bytes)?,
            read_varint(&mut bytes)?,
        ];
        if entry[2] == 0 || (i != 0 && entry[0] == 0) {
            return Err("invalid index entry");
        }
        entries.push(entry);
    }
    if !bytes.is_empty() {
        return Err("trailing index payload");
    }
    Ok(Index { ticks, entries })
}

fn indexed_suffix(transport: &Transport<'_>) -> Vec<u8> {
    let index = parse_index(transport.index.expect("automatic jxli"), 2).unwrap();
    assert_eq!(index.ticks, [1, 24]);
    assert_eq!(index.entries.len(), 2);
    assert_eq!(
        &index.entries[0][1..],
        &[18, 2],
        "11+0+7 ticks, two displayed frames"
    );
    assert_eq!(
        &index.entries[1][1..],
        &[0, 1],
        "last zero-duration frame is displayed"
    );
    assert_eq!(transport.chunks.len(), 4);
    assert_eq!(transport.xml_boxes, 1);
    let header_len = usize::try_from(index.entries[0][0]).unwrap();
    let frame_offset = usize::try_from(
        index.entries[0][0]
            .checked_add(index.entries[1][0])
            .unwrap(),
    )
    .unwrap();
    assert!(header_len > 0 && header_len < transport.chunks[0].len());
    assert_eq!(
        frame_offset,
        transport.chunks[..3]
            .iter()
            .map(|chunk| chunk.len())
            .sum::<usize>(),
        "index offset excludes interleaved XML and all box framing"
    );
    let raw = transport.chunks.concat();
    let data_len = header_len.checked_add(raw.len() - frame_offset).unwrap();
    // Keep the independently parsed Level 10 container prefix. Only the global
    // image header and the actual indexed suffix enter this new jxlc payload.
    let mut suffix = transport.prefix.to_vec();
    suffix.extend_from_slice(&u32::try_from(data_len + 8).unwrap().to_be_bytes());
    suffix.extend_from_slice(b"jxlc");
    suffix.extend_from_slice(&raw[..header_len]);
    suffix.extend_from_slice(&raw[frame_offset..]);
    suffix
}

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let root = env::temp_dir().canonicalize().unwrap();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            "rs-jxl-sequence-{}-{stamp}-{id}",
            std::process::id()
        ));
        assert_eq!(path.parent(), Some(root.as_path()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cleanup {}: {error}", self.0.display());
        }
    }
}
struct Reap(Child);
impl Drop for Reap {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}
struct FailureLog<'a>(&'a str);
impl Drop for FailureLog<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            eprintln!("{}", self.0);
        }
    }
}
fn read_log(path: &Path) -> String {
    let mut bytes = Vec::new();
    File::open(path)
        .unwrap()
        .take(LOG_LIMIT + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes.len() as u64 <= LOG_LIMIT);
    String::from_utf8_lossy(&bytes).into_owned()
}
fn run(command: &mut Command, scratch: &Path, context: &str) -> String {
    let stdout = scratch.join("stdout.log");
    let stderr = scratch.join("stderr.log");
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    let mut child = Reap(
        command
            .spawn()
            .unwrap_or_else(|e| panic!("{context}: spawn: {e}")),
    );
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            start.elapsed() < Duration::from_secs(240),
            "{context}: timeout\n{}\n{}",
            read_log(&stdout),
            read_log(&stderr)
        );
        for path in [&stdout, &stderr] {
            assert!(
                fs::metadata(path).unwrap().len() <= LOG_LIMIT,
                "{context}: log limit"
            );
        }
        thread::sleep(Duration::from_millis(10));
    };
    let log = format!(
        "stdout:\n{}\nstderr:\n{}",
        read_log(&stdout),
        read_log(&stderr)
    );
    assert!(status.success(), "{context}: {status}\n{log}");
    log
}

fn decoders(scratch: &Path) -> Vec<PathBuf> {
    let paths =
        env::var_os("RS_JXL_DJXL").expect("RS_JXL_DJXL must list official 0.12 and 0.13 decoders");
    let paths: Vec<_> = env::split_paths(&paths)
        .map(|p| p.canonicalize().unwrap())
        .collect();
    let mut versions = [false; 2];
    for path in &paths {
        let log = run(
            Command::new(path).arg("--version"),
            scratch,
            "decoder version",
        );
        assert!(log.contains("djxl v"), "{log}");
        versions[0] |= log.contains("djxl v0.12.");
        versions[1] |= log.contains("djxl v0.13.");
    }
    assert_eq!(
        versions,
        [true, true],
        "both official decoder versions required"
    );
    paths
}

fn decode(input: &Path, expected: &[Image], decoders: &[PathBuf], scratch: &Path, context: &str) {
    for (index, decoder) in decoders.iter().enumerate() {
        let output = scratch.join(format!("decoded-{index}.npy"));
        assert!(!output.exists());
        let log = run(
            Command::new(decoder).arg(input).arg(&output).args([
                "--output_frames",
                "--num_threads=1",
                "--quiet",
            ]),
            scratch,
            context,
        );
        let _failure = FailureLog(&log);
        assert!(
            fs::metadata(&output).unwrap().len() <= 64 << 20,
            "NPY output limit"
        );
        let decoded = parse_npy(&fs::read(&output).unwrap()).unwrap();
        assert_samples(&decoded, expected, context);
        fs::remove_file(output).unwrap();
    }
}

fn oracle(test: &str, make: fn() -> Fixture) {
    if let Some(index) = env::var_os(CHILD) {
        let index: usize = index.to_str().unwrap().parse().unwrap();
        let threads = [1, 4][index];
        let bytes = encode_fixture(make(), threads);
        let output = env::var_os(CHILD_OUTPUT).unwrap();
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        return;
    }
    let scratch = Scratch::new();
    let decoders = decoders(&scratch.0);
    let fixture = make();
    let indexed = !fixture.indexed_frames.is_empty();
    for (index, threads) in [1, 4].into_iter().enumerate() {
        let started = Instant::now();
        let input = scratch.0.join(format!("case-{index}.jxl"));
        let context = format!("{test} threads={threads}");
        let log = run(
            Command::new(env::current_exe().unwrap())
                .args([
                    "--exact",
                    test,
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, index.to_string())
                .env(CHILD_OUTPUT, &input)
                .env("RS_JXL_PROFILE", "1"),
            &scratch.0,
            &context,
        );
        let _failure = FailureLog(&log);
        let length = fs::metadata(&input).unwrap().len();
        assert!(length > 0 && length <= OUTPUT_LIMIT as u64);
        let manifest = log
            .lines()
            .find(|line| line.contains("[sequence acceptance]"))
            .expect("child route assertions completed");
        let bytes = fs::read(&input).unwrap();
        let transport = parse_transport(&bytes).unwrap();
        assert_eq!(transport.chunks.len(), fixture.submissions.len());
        assert_eq!(transport.index.is_some(), indexed);
        assert_eq!(transport.xml_boxes, usize::from(indexed));
        decode(&input, &fixture.expected, &decoders, &scratch.0, &context);
        if indexed {
            assert_eq!(fixture.indexed_frames, [0, 3]);
            let suffix = indexed_suffix(&transport);
            let path = scratch.0.join(format!("suffix-{index}.jxl"));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap()
                .write_all(&suffix)
                .unwrap();
            decode(
                &path,
                std::slice::from_ref(fixture.expected.last().unwrap()),
                &decoders,
                &scratch.0,
                &format!("{context} independent indexed suffix"),
            );
        }
        eprintln!(
            "{context}: bytes={length} elapsed={:.3}s {manifest}",
            started.elapsed().as_secs_f64()
        );
    }
    eprintln!(
        "{test}: 2 encodes, {} official pixel comparisons",
        decoders.len() * 2 * if indexed { 2 } else { 1 }
    );
}

#[test]
#[ignore = "official djxl 0.12/0.13 pixel oracle; parent runs centrally"]
fn sequence_mixed_areas_crops_references_and_zero_duration() {
    oracle(
        "sequence_mixed_areas_crops_references_and_zero_duration",
        mixed_fixture,
    );
}

#[test]
#[ignore = "official djxl 0.12/0.13 pixel oracle; parent runs centrally"]
fn sequence_area_frames_with_full_and_subsampled_extras() {
    oracle(
        "sequence_area_frames_with_full_and_subsampled_extras",
        extras_fixture,
    );
}

#[test]
#[ignore = "official djxl 0.12/0.13 pixel oracle; parent runs centrally"]
fn sequence_indexed_displayed_frames_and_independent_suffix() {
    oracle(
        "sequence_indexed_displayed_frames_and_independent_suffix",
        indexed_fixture,
    );
}

#[test]
#[ignore = "official djxl 0.12/0.13 pixel oracle; parent runs centrally"]
fn sequence_f16_special_words_and_crop_layers() {
    oracle("sequence_f16_special_words_and_crop_layers", half_fixture);
}

#[test]
fn mixed_fixture_has_real_area_gate_and_hidden_rgb() {
    let fixture = mixed_fixture();
    assert_eq!(fixture.expected.len(), 2);
    for submission in &fixture.submissions {
        let plan = FrameAreaPlan::new(
            ImageDescriptor::from_image(&submission.image).unwrap(),
            &options(1),
        );
        if submission.mode == FrameProgressMode::FrameAreas {
            // g0 is 128 here: ceil(513/128)^2 = 25, strictly above eight.
            assert_eq!(plan.unwrap().num_groups(), 25);
        } else {
            assert!(plan.is_err(), "tiny crops must take explicit fallback");
        }
    }
    let PixelData::U8(samples) = fixture.expected[0].pixels() else {
        panic!()
    };
    assert_eq!(&samples[..4], &[17, 223, 91, 0]);
}

#[test]
fn extra_and_small_fixtures_select_the_requested_routes() {
    let extras = extras_fixture();
    assert_eq!(extras.expected.len(), 3);
    assert_eq!(extras.buffering, FrameBuffering::StrictAreas);
    for submission in extras.submissions {
        let descriptor = ImageDescriptor::from_image(&submission.image).unwrap();
        assert_eq!(descriptor.channel_count(), 6);
        assert_eq!(descriptor.channel_dimensions(4).unwrap(), (513, 513));
        assert_eq!(descriptor.channel_dimensions(5).unwrap(), (257, 257));
        assert_eq!(
            FrameAreaPlan::new(descriptor, &options(1))
                .unwrap()
                .num_groups(),
            25
        );
        assert_eq!(submission.mode, FrameProgressMode::FrameAreas);
    }
    for (fixture, displayed) in [(indexed_fixture(), 3), (half_fixture(), 2)] {
        assert_eq!(fixture.expected.len(), displayed);
        for submission in fixture.submissions {
            assert_eq!(submission.mode, FrameProgressMode::CurrentFrameBuffered);
            assert!(
                FrameAreaPlan::new(
                    ImageDescriptor::from_image(&submission.image).unwrap(),
                    &options(1)
                )
                .is_err()
            );
        }
    }
}

#[test]
fn index_reader_checks_counts_ticks_varints_and_payload_bounds() {
    let bytes = [2, 0, 0, 0, 1, 0, 0, 0, 24, 17, 18, 2, 0x81, 1, 0, 1];
    assert_eq!(
        parse_index(&bytes, 2).unwrap(),
        Index {
            ticks: [1, 24],
            entries: vec![[17, 18, 2], [129, 0, 1]],
        }
    );
    for end in 0..bytes.len() {
        assert!(parse_index(&bytes[..end], 2).is_err());
    }
    assert!(parse_index(&bytes, 1).is_err());
    let mut trailing = bytes.to_vec();
    trailing.push(0);
    assert!(parse_index(&trailing, 2).is_err());
    let mut invalid = bytes;
    invalid[5..9].fill(0);
    assert!(parse_index(&invalid, 2).is_err());
    assert!(read_varint(&mut &[0x80, 0][..]).is_err());
    assert!(read_varint(&mut &[0xff; 9][..]).is_err());
    let mut maximum = [0xff; 9];
    maximum[8] = 0x7f;
    assert_eq!(
        read_varint(&mut maximum.as_slice()).unwrap(),
        i64::MAX as u64
    );
}

#[test]
fn npy_parser_and_half_words_are_exact_and_bounded() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 1, 1, 2), }\n";
    let mut bytes = b"\x93NUMPY\x01\0".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    let words: [u32; 4] = [0, 0x8000_0000, 0x7fc0_2000, 0xff80_0000];
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
        assert!(parse_npy(&bytes[..end]).is_err());
    }
    bytes.push(0);
    assert!(parse_npy(&bytes).is_err());
    for (half, word) in [
        (0, 0),
        (0x8000, 0x8000_0000),
        (1, 0x3380_0000),
        (0x03ff, 0x387f_c000),
        (0x0400, 0x3880_0000),
        (0x3c00, 0x3f80_0000),
        (0x7c00, 0x7f80_0000),
        (0xfc00, 0xff80_0000),
        (0x7e01, 0x7fc0_2000),
        (0xfc01, 0xff80_2000),
    ] {
        assert_eq!(half_word(half), word);
    }
}
