use super::*;
use crate::fast_lossless::{Backend, Limits, encode};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

pub(in crate::fast_lossless) struct Generated {
    pub info: FrameInfo,
    palette: bool,
    calls: Mutex<Vec<Region>>,
    info_calls: AtomicUsize,
    live: AtomicUsize,
    peak: AtomicUsize,
    releases: AtomicUsize,
    fault: AtomicUsize,
}

impl Generated {
    pub fn new(width: usize, height: usize, channels: u8, depth: u8, palette: bool) -> Self {
        Self {
            info: FrameInfo {
                width,
                height,
                channels,
                bit_depth: depth,
                big_endian: true,
            },
            palette,
            calls: Mutex::new(Vec::new()),
            info_calls: AtomicUsize::new(0),
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            releases: AtomicUsize::new(0),
            fault: AtomicUsize::new(0),
        }
    }

    fn pixels(&self, region: Region) -> (Vec<u8>, usize) {
        let storage = if self.info.bit_depth <= 8 { 1 } else { 2 };
        let channels = usize::from(self.info.channels);
        let row = region.width * channels * storage;
        let stride = row + 3;
        let mut bytes = vec![0x79; (region.height - 1) * stride + row];
        let colors = [
            [0, 0, 0, 0],
            [13, 227, 7, 255],
            [199, 31, 241, 0],
            [63, 171, 29, 50],
        ];
        for y in 0..region.height {
            for x in 0..region.width {
                let (gx, gy) = (x + region.x, y + region.y);
                let color = colors[(gx / 3 + gy / 2) % colors.len()];
                for (c, &palette_value) in color.iter().take(channels).enumerate() {
                    let value = if self.palette {
                        palette_value
                    } else if c == 3 && gx % 3 == 0 {
                        0
                    } else {
                        ((gx * 8191 + gy * 971 + c * 211 + gx * gy)
                            & ((1 << self.info.bit_depth) - 1)) as u16
                    };
                    let at = y * stride + (x * channels + c) * storage;
                    if storage == 1 {
                        bytes[at] = value as u8;
                    } else {
                        bytes[at..at + 2].copy_from_slice(&value.to_be_bytes());
                    }
                }
            }
        }
        (bytes, stride)
    }

    // Test oracle only: production input allocates just each requested region.
    pub fn expected_pixels(&self) -> (Vec<u8>, usize) {
        self.pixels(Region {
            x: 0,
            y: 0,
            width: self.info.width,
            height: self.info.height,
        })
    }

    pub fn image<'a>(&self, pixels: &'a [u8], row_stride: usize) -> ImageView<'a> {
        ImageView {
            pixels,
            width: self.info.width,
            height: self.info.height,
            row_stride,
            channels: self.info.channels,
            bit_depth: self.info.bit_depth,
            big_endian: self.info.big_endian,
        }
    }

    fn assert_released(&self) {
        assert_eq!(self.live.load(Ordering::SeqCst), 0);
        assert_eq!(
            self.releases.load(Ordering::SeqCst),
            self.calls.lock().unwrap().len()
        );
    }
}

pub(in crate::fast_lossless) struct GeneratedLease<'a> {
    source: &'a Generated,
    bytes: Vec<u8>,
    stride: usize,
    panic: bool,
}
impl RegionBuffer for GeneratedLease<'_> {
    fn bytes(&self) -> &[u8] {
        assert!(!self.panic, "injected input lease panic");
        &self.bytes
    }
    fn row_stride(&self) -> usize {
        self.stride
    }
}
impl Drop for GeneratedLease<'_> {
    fn drop(&mut self) {
        self.source.live.fetch_sub(1, Ordering::SeqCst);
        self.source.releases.fetch_add(1, Ordering::SeqCst);
    }
}
impl RegionSource for Generated {
    type Buffer<'a> = GeneratedLease<'a>;
    fn info(&self) -> FrameInfo {
        self.info_calls.fetch_add(1, Ordering::SeqCst);
        self.info
    }
    fn get_region(&self, region: Region) -> Result<Self::Buffer<'_>> {
        let fault = self.fault.load(Ordering::SeqCst);
        if fault == 1 {
            return Err(Error::InvalidInput("injected get failure".into()));
        }
        assert_eq!((region.x % 256, region.y % 256), (0, 0));
        assert_eq!(region.width, (self.info.width - region.x).min(256));
        assert_eq!(region.height, (self.info.height - region.y).min(256));
        let (mut bytes, stride) = self.pixels(region);
        if fault == 2 {
            bytes.pop();
        }
        if fault == 4 {
            bytes[0] = 255;
        }
        if fault == 5 {
            bytes[0] ^= 1;
        }
        self.calls.lock().unwrap().push(region);
        let live = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(live, Ordering::SeqCst);
        Ok(GeneratedLease {
            source: self,
            bytes,
            stride,
            panic: fault == 3,
        })
    }
}

pub(in crate::fast_lossless) struct MemoryOutput {
    pub bytes: Vec<u8>,
    buffer: Vec<u8>,
    position: u64,
    finalized: u64,
    active: bool,
    gets: usize,
    releases: Vec<usize>,
    seeks: Vec<u64>,
    finalizations: usize,
    fault: u8,
}
impl MemoryOutput {
    pub fn new(chunk: usize) -> Self {
        Self {
            bytes: Vec::new(),
            buffer: vec![0; chunk],
            position: 0,
            finalized: 0,
            active: false,
            gets: 0,
            releases: Vec::new(),
            seeks: Vec::new(),
            finalizations: 0,
            fault: 0,
        }
    }
    fn balanced(&self) {
        assert!(!self.active);
        assert_eq!(self.gets, self.releases.len());
    }
}
impl OutputProcessor for MemoryOutput {
    fn position(&self) -> u64 {
        self.position
    }
    fn get_buffer(&mut self, suggested: usize) -> Result<&mut [u8]> {
        assert!(!self.active);
        assert_ne!(suggested, 0);
        if self.fault == 1 {
            return Err(Error::Encoding("injected get failure".into()));
        }
        self.active = true;
        self.gets += 1;
        Ok(&mut self.buffer)
    }
    fn release_buffer(&mut self, written: usize) -> Result<()> {
        assert!(self.active);
        assert!(written <= self.buffer.len());
        self.active = false;
        self.releases.push(written);
        if self.fault == 2 {
            return Err(Error::Encoding("injected release failure".into()));
        }
        if written != 0 {
            let start = usize::try_from(self.position).unwrap();
            self.bytes.resize(self.bytes.len().max(start + written), 0);
            self.bytes[start..start + written].copy_from_slice(&self.buffer[..written]);
            if self.fault != 4 {
                self.position += written as u64;
            }
        }
        Ok(())
    }
    fn seek(&mut self, position: u64) -> Result<()> {
        assert!(!self.active);
        assert!(position >= self.finalized);
        if self.fault == 3 {
            return Err(Error::Encoding("injected seek failure".into()));
        }
        self.seeks.push(position);
        if self.fault != 5 {
            self.position = position;
        }
        Ok(())
    }
    fn set_finalized_position(&mut self, position: u64) -> Result<()> {
        assert!(!self.active);
        assert!(position >= self.finalized && position <= self.bytes.len() as u64);
        self.finalized = position;
        self.finalizations += 1;
        Ok(())
    }
}

fn options() -> Options {
    Options {
        backend: Backend::Scalar,
        oneshot: false,
        ..Options::default()
    }
}
fn encoded(source: &Generated) -> EncodedFrame {
    prepare_frame(source, &options())
        .unwrap()
        .process()
        .unwrap()
        .prepare_header(true, true)
        .unwrap()
}

#[test]
fn source_region_geometry_sampling_and_release_order() {
    for effort in [0, 2, 63, 64, 127] {
        let source = Generated::new(513, 257, 3, 14, false);
        let options = Options {
            effort,
            ..options()
        };
        let prepared = prepare_frame(&source, &options).unwrap();
        assert_eq!(prepared.info(), source.info);
        assert_eq!(source.info_calls.load(Ordering::SeqCst), 1);
        source.assert_released();
        let sampled = source.calls.lock().unwrap().clone();
        if effort < 64 {
            assert_eq!(
                sampled,
                [Region {
                    x: 256,
                    y: 0,
                    width: 256,
                    height: 256
                }]
            );
        } else {
            assert_eq!(sampled.len(), 6);
        }
        let frame = prepared
            .process()
            .unwrap()
            .prepare_header(true, true)
            .unwrap();
        source.assert_released();
        let calls = source.calls.lock().unwrap();
        assert_eq!(calls.len(), sampled.len() + 6);
        assert_eq!(
            calls.last(),
            Some(&Region {
                x: 512,
                y: 256,
                width: 1,
                height: 1
            })
        );
        assert_eq!(source.peak.load(Ordering::SeqCst), 1);
        let (pixels, stride) = source.expected_pixels();
        assert_eq!(
            frame.into_bytes().unwrap(),
            encode(source.image(&pixels, stride), &options).unwrap()
        );
    }
}

#[test]
fn callback_palette_and_borrowed_input_have_identical_dictionary_output() {
    for channels in 1..=4 {
        let source = Generated::new(257, 5, channels, 8, true);
        let options = Options {
            oneshot: true,
            ..options()
        };
        let prepared = prepare_frame(&source, &options).unwrap();
        assert!(prepared.palette.is_some());
        assert_eq!(source.calls.lock().unwrap().len(), 4);
        source.assert_released();
        let result = prepared
            .process()
            .unwrap()
            .prepare_header(true, true)
            .unwrap()
            .into_bytes()
            .unwrap();
        source.assert_released();
        assert_eq!(source.calls.lock().unwrap().len(), 6);
        let (pixels, stride) = source.expected_pixels();
        assert_eq!(
            result,
            encode(source.image(&pixels, stride), &options).unwrap()
        );
    }
}

#[test]
fn leases_drop_on_invalid_data_errors_and_unwind() {
    for fault in 1..=4 {
        let source = Generated::new(17, 3, 4, 7, false);
        source.fault.store(fault, Ordering::SeqCst);
        let result = std::panic::catch_unwind(|| prepare_frame(&source, &options()));
        if fault == 3 {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        source.assert_released();
    }
    let source = Generated::new(257, 3, 4, 16, false);
    let prepared = prepare_frame(&source, &options()).unwrap();
    source.fault.store(2, Ordering::SeqCst);
    assert!(prepared.process().is_err());
    source.assert_released();
}

#[test]
fn palette_source_changes_are_rejected_instead_of_aliasing_indices() {
    let source = Generated::new(17, 3, 4, 8, true);
    let prepared = prepare_frame(
        &source,
        &Options {
            oneshot: true,
            ..options()
        },
    )
    .unwrap();
    assert!(prepared.palette.is_some());
    source.fault.store(5, Ordering::SeqCst);
    assert!(prepared.process().is_err());
    source.assert_released();
}

#[test]
fn incremental_output_handles_empty_sections_tiny_buffers_and_eof() {
    let source = Generated::new(257, 3, 4, 16, false);
    let expected = encoded(&source).into_bytes().unwrap();
    for chunk in [1, 7, 31, 32, 33, 4096] {
        let mut frame = encoded(&source);
        assert_eq!(frame.output_size(), expected.len());
        assert!(frame.write_output(&mut []).is_err());
        let mut bytes = Vec::new();
        let mut buffer = vec![0; chunk];
        while frame.remaining_size() != 0 {
            let n = frame.write_output(&mut buffer).unwrap();
            assert!((1..=chunk).contains(&n));
            bytes.extend_from_slice(&buffer[..n]);
            assert_eq!(frame.remaining_size(), expected.len() - bytes.len());
        }
        assert_eq!(frame.write_output(&mut []).unwrap(), 0);
        assert_eq!(bytes, expected);
        assert_eq!(frame.into_bytes().unwrap(), expected);
    }
}

#[test]
fn output_callbacks_are_balanced_and_finalization_belongs_to_caller() {
    let source = Generated::new(257, 3, 2, 8, false);
    let mut frame = encoded(&source);
    let mut output = MemoryOutput::new(13);
    let mut initial = [0; 7];
    assert_eq!(frame.write_output(&mut initial).unwrap(), initial.len());
    write_all(&mut output, &initial).unwrap();
    frame.write_to(&mut output).unwrap();
    output.balanced();
    assert_eq!(output.bytes, frame.into_bytes().unwrap());
    assert!(output.seeks.is_empty());
    assert_eq!(output.finalizations, 0);
    output.set_finalized_position(output.position()).unwrap();
    assert_eq!(output.finalized, output.bytes.len() as u64);
}

#[test]
fn callback_stop_and_failure_are_terminal_and_never_double_release() {
    let source = Generated::new(17, 3, 1, 8, false);
    for (chunk, fault) in [(0, 0), (17, 1), (17, 2), (17, 4)] {
        let mut frame = encoded(&source);
        let mut output = MemoryOutput::new(chunk);
        output.fault = fault;
        assert!(frame.write_to(&mut output).is_err());
        output.balanced();
        assert_eq!(output.finalizations, 0);
        if chunk == 0 {
            assert_eq!(output.releases, [0]);
        }
        output.fault = 0;
        assert!(frame.write_to(&mut output).is_err());
        assert!(frame.write_output(&mut [0; 1]).is_err());
    }
    let mut output = MemoryOutput::new(17);
    output.position = u64::MAX - 1;
    assert!(encoded(&source).write_to(&mut output).is_err());
    assert_eq!(output.releases, [0]);
    assert!(output.bytes.is_empty());
    output.balanced();
}

#[test]
fn seek_streaming_backpatches_relative_to_start_without_finalizing_outer_header() {
    let source = Generated::new(513, 7, 4, 16, false);
    let mut output = MemoryOutput::new(17);
    write_all(&mut output, b"pending-box").unwrap();
    let start = output.position();
    let count = prepare_frame(&source, &options())
        .unwrap()
        .process_to(&mut output, true, true)
        .unwrap();
    output.balanced();
    source.assert_released();
    assert_eq!(output.position(), start + count as u64);
    assert_eq!(&output.bytes[..start as usize], b"pending-box");
    assert_eq!(
        &output.bytes[start as usize..start as usize + 2],
        &[0xff, 0x0a]
    );
    assert_eq!(output.seeks.len(), 3);
    assert_eq!(output.seeks[1], start);
    assert_eq!(output.finalizations, 0);
    output.seek(0).unwrap();
    write_all(&mut output, b"patched-box").unwrap();
    output.seek(start + count as u64).unwrap();
    output.set_finalized_position(output.position()).unwrap();
}

#[test]
fn single_group_streaming_needs_no_seek_and_matches_incremental_bytes() {
    let source = Generated::new(17, 3, 4, 14, false);
    let mut output = MemoryOutput::new(7);
    output.fault = 3;
    let size = prepare_frame(&source, &options())
        .unwrap()
        .process_to(&mut output, true, true)
        .unwrap();
    assert_eq!(size, output.bytes.len());
    assert_eq!(output.bytes, encoded(&source).into_bytes().unwrap());
    output.balanced();
    assert!(output.seeks.is_empty());
}

#[test]
fn streaming_seek_failure_and_output_limits_do_not_finalize() {
    let source = Generated::new(257, 3, 4, 16, false);
    for fault in [3, 5] {
        let mut output = MemoryOutput::new(17);
        output.fault = fault;
        assert!(
            prepare_frame(&source, &options())
                .unwrap()
                .process_to(&mut output, false, true)
                .is_err()
        );
        output.balanced();
        assert_eq!(output.finalizations, 0);
    }
    let prepared = prepare_frame(&source, &options()).unwrap();
    let (_, offset) = reservation(prepared.global.bits_written().div_ceil(8), 1, 2).unwrap();
    drop(prepared);
    for cap in [offset - 1, offset + 1] {
        let limited = Options {
            limits: Limits {
                max_output_bytes: cap,
                ..Limits::default()
            },
            ..options()
        };
        let mut output = MemoryOutput::new(17);
        assert!(
            prepare_frame(&source, &limited)
                .unwrap()
                .process_to(&mut output, false, true)
                .is_err()
        );
        assert_eq!(output.finalizations, 0);
        assert!(output.bytes.len() <= cap);
        output.balanced();
    }
}

#[test]
fn source_reservation_keeps_global_toc_bucket_stable_at_boundaries() {
    for original in [0, 1020, 1023, 1024, 17407, 17408, 4211711, 4211712] {
        for ac in [2usize, 17, 1024] {
            let dc = ac.div_ceil(64);
            let (minimum, offset) = reservation(original, dc, ac).unwrap();
            for ac_size in [0, 1023, 1024, 17408, 4211711] {
                let mut sizes = vec![ac_size; 2 + dc + ac];
                sizes[..2 + dc].fill(0);
                sizes[0] = minimum;
                for (channels, last) in [(1, true), (4, true), (4, false)] {
                    let mut header = BitWriter::new(1 << 20);
                    header::frame(&mut header, channels, last, &sizes).unwrap();
                    let padded = offset - header.bits_written() / 8;
                    assert!(padded >= original);
                    assert_eq!(toc_bits(padded).unwrap(), toc_bits(minimum).unwrap());
                    sizes[0] = padded;
                    let mut actual = BitWriter::new(1 << 20);
                    header::frame(&mut actual, channels, last, &sizes).unwrap();
                    assert_eq!(actual.bits_written(), header.bits_written());
                }
            }
        }
    }
    assert!(reservation(usize::MAX, 1, 2).is_err());
    assert!(reservation(1, usize::MAX, 2).is_err());
    assert!(reservation(1, 1, usize::MAX).is_err());
}

#[test]
fn pool_processing_preserves_order_and_bounds_live_regions() {
    let source = Generated::new(4097, 3, 4, 16, false);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let expected = encoded(&source).into_bytes().unwrap();
    let frame = prepare_frame(&source, &options())
        .unwrap()
        .process_with_pool(&pool)
        .unwrap()
        .prepare_header(true, true)
        .unwrap();
    assert_eq!(frame.into_bytes().unwrap(), expected);
    let mut serial = MemoryOutput::new(73);
    prepare_frame(&source, &options())
        .unwrap()
        .process_to(&mut serial, true, true)
        .unwrap();
    let mut parallel = MemoryOutput::new(73);
    prepare_frame(&source, &options())
        .unwrap()
        .process_to_with_pool(&mut parallel, true, true, &pool)
        .unwrap();
    assert_eq!(serial.bytes, parallel.bytes);
    source.assert_released();
    assert!(source.peak.load(Ordering::SeqCst) <= 16);
    let prepared = prepare_frame(&source, &options()).unwrap();
    source.fault.store(2, Ordering::SeqCst);
    assert!(prepared.process_with_pool(&pool).is_err());
    source.assert_released();
}

#[test]
fn borrowed_source_checks_full_stride_and_region_overflow() {
    let pixels = vec![0; 4096];
    let image = ImageView {
        pixels: &pixels,
        width: 513,
        height: 2,
        row_stride: 256,
        channels: 1,
        bit_depth: 8,
        big_endian: false,
    };
    assert!(prepare_frame(&image, &options()).is_err());
    for region in [
        Region {
            x: usize::MAX,
            y: 0,
            width: 1,
            height: 1,
        },
        Region {
            x: 0,
            y: 0,
            width: 0,
            height: 1,
        },
    ] {
        assert!(image.get_region(region).is_err());
    }
}

#[test]
fn invalid_metadata_and_group_budget_fail_before_get_region() {
    for (width, depth, cap) in [(0, 8, 10), (1, 17, 10), (257, 8, 1)] {
        let source = Generated::new(width, 1, 1, depth, false);
        let options = Options {
            limits: Limits {
                max_groups: cap,
                ..Limits::default()
            },
            ..options()
        };
        assert!(prepare_frame(&source, &options).is_err());
        assert!(source.calls.lock().unwrap().is_empty());
    }
}

#[test]
fn prepared_header_reuse_does_not_reacquire_or_reencode_input() {
    let source = Generated::new(257, 3, 4, 8, false);
    let mut frame = encoded(&source);
    let requests = source.calls.lock().unwrap().len();
    let (pixels, stride) = source.expected_pixels();
    let image = source.image(&pixels, stride);
    for last in [false, true] {
        frame.reset_header(true, true).unwrap();
        assert_eq!(frame.write_output(&mut [0; 7]).unwrap(), 7);
        frame.reset_header(false, last).unwrap();
        assert_eq!(frame.remaining_size(), frame.output_size());
        let expected = crate::fast_lossless::encode_frame(image, &options(), last).unwrap();
        let mut output = MemoryOutput::new(31);
        frame.write_to(&mut output).unwrap();
        assert_eq!(output.bytes, expected);
    }
    assert_eq!(source.calls.lock().unwrap().len(), requests);
    source.assert_released();
}
