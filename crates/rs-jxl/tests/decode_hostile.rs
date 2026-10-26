//! Small deterministic hostile-input corpus; no external files or fuzz runtime.
//! Each batch runs in a killable process: panics, aborts and timeouts fail the
//! parent test. Invalid bitstreams may still decode if the changed bits are legal.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use rs_jxl::container::ChunkedWriter;
use rs_jxl::decode::{
    DecodeLimits, DecodeOptions, DecodedImage, decode, decode_frame, inspect, read_basic_info,
    reconstruct_jpeg,
};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::jpeg::{TranscodeOptions, transcode};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::{EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout, encode};

const SEED: u64 = 0x83e1_96d4_02a7_5b19;
const CHILD_ENV: &str = "RS_JXL_HOSTILE_BATCH";
const DEADLINE: Duration = Duration::from_secs(45);

fn bounded_options() -> DecodeOptions {
    DecodeOptions {
        limits: DecodeLimits {
            max_input_bytes: 32 * 1024,
            max_output_bytes: 256 * 1024,
            max_metadata_bytes: 16 * 1024,
            max_pixels: 4096,
            max_samples: 16 * 1024,
            max_frames: 4,
        },
        ..Default::default()
    }
}

fn isolated(batch: &str) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "hostile_worker", "--ignored", "--nocapture"])
        .env(CHILD_ENV, batch)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("start isolated hostile-input test");
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                assert!(status.success(), "batch {batch}, seed {SEED:#x}: {status}");
                return;
            }
            Ok(None) if start.elapsed() < DEADLINE => {
                std::thread::sleep(Duration::from_millis(10));
            }
            result => {
                // Do not detach a stuck decoder thread or accept a timeout as OK.
                let killed = child.kill();
                let reaped = child.wait();
                panic!(
                    "batch {batch}, seed {SEED:#x}: deadline/wait failure {result:?}; \
                     kill={killed:?}, wait={reaped:?}"
                );
            }
        }
    }
}

#[test]
fn bounded_truncations_and_seeded_bitstream_mutations() {
    isolated("pixels");
}

#[test]
fn malformed_container_lengths_and_brotli_budgets() {
    isolated("containers");
}

#[test]
fn original_jpeg_reconstruction_rejects_structural_corruption() {
    isolated("jpeg");
}

#[test]
fn decoder_limits_and_invalid_options_have_distinct_errors() {
    isolated("budgets");
}

#[test]
#[ignore = "subprocess entry point; run the four parent tests instead"]
fn hostile_worker() {
    match std::env::var(CHILD_ENV).as_deref() {
        Ok("pixels") => pixel_corpus(),
        Ok("containers") => container_corpus(),
        Ok("jpeg") => jpeg_corpus(),
        Ok("budgets") => budgets(),
        other => panic!("missing/invalid isolated batch: {other:?}"),
    }
}

fn checkpoint(label: &str, index: usize) {
    // Inherited stderr cannot fill an unread pipe. The last line identifies a
    // stuck case even when the child must be forcibly killed.
    eprintln!("hostile seed={SEED:#x} batch={label} case={index}");
}

fn pixel_fixture() -> (Image, Vec<u8>) {
    let pixels = (0..16 * 8 * 4)
        .map(|i| ((i * 37 + i / 7) & 255) as u8)
        .collect();
    let image = Image::new(16, 8, PixelLayout::Rgba, PixelData::U8(pixels)).unwrap();
    // Image defaults conservatively require a Level 10 container. This fixture
    // uses U8 + Zero prediction with no transforms, so 16-bit intermediates are
    // sufficient and a bare Level 5 codestream is explicitly safe here.
    let mut metadata = image.metadata().clone();
    metadata.modular_16_bit_buffer_sufficient = true;
    let image = image.with_metadata(metadata, vec![]).unwrap();
    let options = EncodeOptions {
        modular: ModularOptions {
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            rct: None,
            lz77: false,
            entropy: Coding::Prefix,
            ..Default::default()
        },
        ..Default::default()
    };
    let bytes = encode(&image, &options).unwrap();
    assert!(bytes.starts_with(&[0xff, 0x0a]));
    assert!(bytes.len() < 4096);
    (image, bytes)
}

fn container(codestream: &[u8]) -> Vec<u8> {
    let mut writer = ChunkedWriter::new(Vec::new(), 5, 32 * 1024).unwrap();
    let split = codestream.len() / 2;
    writer.write_chunk(&codestream[..split], false).unwrap();
    writer.write_chunk(&codestream[split..], true).unwrap();
    writer.finish().unwrap()
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn cuts(len: usize) -> Vec<usize> {
    let mut cuts: Vec<_> = (0..len.min(24))
        .chain(len.saturating_sub(24)..len)
        .chain((0..len).step_by((len / 64).max(1)))
        .collect();
    cuts.sort_unstable();
    cuts.dedup();
    assert!(cuts.len() <= 176);
    cuts
}

fn expected_error<T>(result: rs_jxl::Result<T>) {
    match result {
        Err(error @ (Error::Decoding(_) | Error::Limit(_) | Error::Io(_))) => {
            assert!(!error.to_string().is_empty());
        }
        Err(other) => panic!("corrupt input misclassified: {other}"),
        Ok(_) => panic!("accepted deliberately malformed input"),
    }
}

fn limit<T>(result: rs_jxl::Result<T>) {
    match result {
        Err(Error::Limit(reason)) => assert!(!reason.is_empty()),
        Err(other) => panic!("budget error misclassified: {other}"),
        Ok(_) => panic!("accepted a request exceeding its budget"),
    }
}

fn successful_decode_is_bounded(image: &DecodedImage, input_len: usize, options: &DecodeOptions) {
    assert!(image.complete);
    assert!(image.bytes_read <= input_len);
    assert!(!image.frames.is_empty());
    assert!(image.frames.len() <= options.limits.max_frames);
    let mut bytes = 0usize;
    for frame in image.frames.iter().chain(image.preview.iter()) {
        assert!(frame.complete);
        let count = frame.width.checked_mul(frame.height).unwrap();
        assert!(count <= options.limits.max_pixels);
        assert_eq!(frame.pixels.len(), count * frame.layout.channels());
        assert_eq!(frame.extra_planes.len(), image.info.extra_channels.len());
        for plane in std::iter::once(&frame.pixels).chain(&frame.extra_planes) {
            if !std::ptr::eq(plane, &frame.pixels) {
                assert_eq!(plane.len(), count);
            }
            bytes = bytes
                .checked_add(plane.len() * usize::from(plane.bits()) / 8)
                .unwrap();
        }
    }
    assert!(bytes <= options.limits.max_output_bytes);
    assert!(
        image.boxes.iter().map(|b| b.data.len()).sum::<usize>()
            <= options.limits.max_metadata_bytes
    );
}

fn probe(bytes: &[u8], options: &DecodeOptions) {
    match decode(bytes, options) {
        Ok(image) => successful_decode_is_bounded(&image, bytes.len(), options),
        // Random mutations can produce unsupported but otherwise parseable input.
        Err(Error::InvalidInput(message)) => assert!(!message.is_empty()),
        Err(error) => expected_error::<()>(Err(error)),
    }
    // Header inspection is NOT an integrity check of the entropy-coded pixels.
    if let Ok(info) = read_basic_info(bytes, options) {
        assert!(info.size.0.checked_mul(info.size.1).unwrap() <= options.limits.max_pixels);
    }
    if let Ok(info) = inspect(bytes, options) {
        assert!(info.frames.len() <= options.limits.max_frames);
        assert!(info.bytes_read <= bytes.len());
    }
}

fn pixel_corpus() {
    let (source, raw) = pixel_fixture();
    let options = bounded_options();
    for bytes in [&raw, &container(&raw)] {
        assert_eq!(
            decode(bytes, &options).unwrap().frames[0].pixels,
            *source.pixels()
        );
        for length in cuts(bytes.len()) {
            checkpoint("pixel-prefix", length);
            expected_error(decode(&bytes[..length], &options));
        }
        let mut rng = SEED;
        for case in 0..64 {
            let mut mutated = bytes.clone();
            let index = (next(&mut rng) % mutated.len() as u64) as usize;
            mutated[index] ^= 1 << (next(&mut rng) % 8);
            checkpoint("pixel-bitflip", case);
            probe(&mutated, &options);
        }
    }
    let mut rng = SEED;
    for case in 0..64 {
        let len = (next(&mut rng) % 255 + 2) as usize;
        let mut bytes: Vec<_> = (0..len).map(|_| next(&mut rng) as u8).collect();
        if case % 2 == 0 {
            bytes[..2].copy_from_slice(&[0xff, 0x0a]);
        }
        checkpoint("random-bytes", case);
        probe(&bytes, &options);
        match reconstruct_jpeg(&bytes, &options) {
            Ok(jpeg) => {
                assert!(jpeg.starts_with(&[0xff, 0xd8]));
                assert!(jpeg.len() <= options.limits.max_output_bytes);
            }
            Err(error) => expected_error::<()>(Err(error)),
        }
    }
}

fn append_box(bytes: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    bytes.extend(u32::try_from(payload.len() + 8).unwrap().to_be_bytes());
    bytes.extend(kind);
    bytes.extend(payload);
}

fn container_corpus() {
    let (source, raw) = pixel_fixture();
    let good = container(&raw);
    let options = bounded_options();
    // Known-invalid short/extended sizes and a truncated compressed metadata box.
    let mut tails = Vec::new();
    for size in [2u32, 3, 4, 5, 6, 7, 20, u32::MAX] {
        let mut tail = size.to_be_bytes().to_vec();
        tail.extend(b"xml ");
        tails.push(tail);
    }
    // Both reference decoders treat extended zero as an EOF-sized box.
    // Do not classify that accepted representation as malformed.
    for size in [8u64, 15, u64::MAX] {
        let mut tail = b"\0\0\0\x01xml ".to_vec();
        tail.extend(size.to_be_bytes());
        tails.push(tail);
    }
    tails.push(b"\0\0\0\x10brobxml \xff\xff\xff\xff".to_vec());
    for (case, tail) in tails.into_iter().enumerate() {
        checkpoint("container-length", case);
        let mut bytes = good.clone();
        bytes.extend(tail);
        expected_error(decode(&bytes, &options));
        expected_error(inspect(&bytes, &options));
        // Selecting a completed frame intentionally need not validate later boxes.
        let selected = decode_frame(&bytes, 0, &options).unwrap();
        assert!(!selected.complete);
        assert_eq!(selected.frames[0].pixels, *source.pixels());
    }
    let mut unknown = good.clone();
    append_box(&mut unknown, b"test", b"opaque valid box");
    assert_eq!(
        decode(&unknown, &options).unwrap().frames[0].pixels,
        *source.pixels()
    );
    let mut open_ended = good.clone();
    open_ended.extend(b"\0\0\0\x01xml ");
    open_ended.extend(0u64.to_be_bytes());
    open_ended.extend(b"metadata through EOF");
    let decoded = decode(&open_ended, &options).unwrap();
    assert_eq!(decoded.frames[0].pixels, *source.pixels());
    assert_eq!(decoded.boxes[0].data, b"metadata through EOF");

    let mut payload = b"xml ".to_vec();
    let expanded = vec![b'x'; 8192];
    {
        let mut compressor = brotli::CompressorWriter::new(&mut payload, 4096, 3, 18);
        compressor.write_all(&expanded).unwrap();
    }
    let mut compressed = good;
    append_box(&mut compressed, b"brob", &payload);
    let mut options = options;
    options.limits.max_metadata_bytes = expanded.len();
    let decoded = decode(&compressed, &options).unwrap();
    assert_eq!(decoded.boxes.len(), 1);
    assert_eq!(decoded.boxes[0].data, expanded);
    options.limits.max_metadata_bytes -= 1;
    limit(decode(&compressed, &options));
    limit(inspect(&compressed, &options));
}

fn marker(bytes: &mut Vec<u8>, code: u8, payload: &[u8]) {
    bytes.extend([0xff, code]);
    bytes.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    bytes.extend(payload);
}

fn jpeg_fixture() -> Vec<u8> {
    // A single grayscale 8x8 block with zero DC/AC. Defined here rather than
    // loading private data; noncanonical padding and trailing bytes are preserved.
    let mut bytes = vec![0xff, 0xd8];
    let mut quant = vec![0];
    quant.extend([1; 64]);
    marker(&mut bytes, 0xdb, &quant);
    marker(&mut bytes, 0xc0, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
    let mut huffman = Vec::new();
    for selector in [0, 16] {
        huffman.extend([selector, 1]);
        huffman.extend([0; 15]);
        huffman.push(0);
    }
    marker(&mut bytes, 0xc4, &huffman);
    marker(&mut bytes, 0xda, &[1, 1, 0, 0, 63, 0]);
    bytes.extend([0x2a, 0xff, 0xd9, 0x52, 0x53]);
    bytes
}

fn box_range(bytes: &[u8], kind: &[u8; 4]) -> std::ops::Range<usize> {
    let mut offset = 0;
    while offset < bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && size <= bytes.len() - offset);
        if &bytes[offset + 4..offset + 8] == kind {
            return offset..offset + size;
        }
        offset += size;
    }
    panic!("fixture missing {kind:?}");
}

fn jpeg_corpus() {
    let original = jpeg_fixture();
    let good = transcode(&original, &TranscodeOptions::default()).unwrap();
    let mut options = bounded_options();
    options.allow_partial = true; // This must never permit incomplete JPEG bytes.
    assert_eq!(reconstruct_jpeg(&good, &options).unwrap(), original);
    for length in cuts(good.len()) {
        checkpoint("jpeg-prefix", length);
        expected_error(reconstruct_jpeg(&good[..length], &options));
    }
    let range = box_range(&good, b"jbrd");
    let mut missing = good.clone();
    missing.drain(range.clone());
    expected_error(reconstruct_jpeg(&missing, &options));
    let mut duplicate = good.clone();
    duplicate.extend_from_slice(&good[range.clone()]);
    expected_error(reconstruct_jpeg(&duplicate, &options));
    let mut rng = SEED;
    let mut rejected = 0;
    for case in 0..64 {
        let mut mutated = good.clone();
        let index = range.start + 8 + (next(&mut rng) % (range.len() - 8) as u64) as usize;
        mutated[index] ^= 1 << (next(&mut rng) % 8);
        checkpoint("jbrd-bitflip", case);
        match reconstruct_jpeg(&mutated, &options) {
            Ok(jpeg) => {
                // Legal changed reconstruction metadata need not equal the input.
                assert!(jpeg.starts_with(&[0xff, 0xd8]));
                assert!(jpeg.len() <= options.limits.max_output_bytes);
            }
            Err(error) => {
                expected_error::<()>(Err(error));
                rejected += 1;
            }
        }
    }
    assert!(rejected > 0, "mutation corpus exercised no rejection paths");
    // Structural damage must be fatal even if all coefficient bytes are intact.
    for size in [2u32, 7, u32::MAX] {
        let mut malformed = good.clone();
        malformed[range.start..range.start + 4].copy_from_slice(&size.to_be_bytes());
        expected_error(reconstruct_jpeg(&malformed, &options));
    }
}

fn budgets() {
    let (_, raw) = pixel_fixture();
    let original = jpeg_fixture();
    let jpeg = transcode(&original, &TranscodeOptions::default()).unwrap();
    for bytes in [&raw, &jpeg] {
        let mut options = bounded_options();
        options.limits.max_input_bytes = bytes.len() - 1;
        limit(decode(bytes, &options));
        limit(decode_frame(bytes, 0, &options));
        limit(read_basic_info(bytes, &options));
        limit(inspect(bytes, &options));
        limit(reconstruct_jpeg(bytes, &options));
    }
    for case in 0..4 {
        checkpoint("budget", case);
        let mut options = bounded_options();
        match case {
            0 => options.limits.max_pixels = 1,
            1 => options.limits.max_samples = 1,
            2 => options.limits.max_output_bytes = 1,
            _ => options.limits.max_frames = 0,
        }
        limit(decode(&raw, &options));
        limit(decode_frame(&raw, 0, &options));
        limit(reconstruct_jpeg(&jpeg, &options));
    }
    for threads in [0, 257] {
        let options = DecodeOptions {
            threads,
            ..bounded_options()
        };
        assert!(matches!(
            decode(&raw, &options),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            reconstruct_jpeg(&jpeg, &options),
            Err(Error::InvalidInput(_))
        ));
    }
    let options = bounded_options();
    let decoded = decode(&raw, &options).unwrap();
    successful_decode_is_bounded(&decoded, raw.len(), &options);
    assert_eq!(reconstruct_jpeg(&jpeg, &options).unwrap(), original);
}
