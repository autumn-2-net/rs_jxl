//! Development-only official decoder oracle. No subprocess is used by the encoder.
//! Parent: cargo test -p rs-jxl fast_lossless::reference_tests -- --ignored --nocapture

use super::*;
use std::env;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let root = env::temp_dir().canonicalize().unwrap();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = root.join(format!("rs-jxl-fjxl-oracle-{}-{stamp}", std::process::id()));
        assert_eq!(path.parent(), Some(root.as_path()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Reap(Child);
impl Drop for Reap {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn run(command: &mut Command, scratch: &Path) {
    let stdout = scratch.join("stdout.txt");
    let stderr = scratch.join("stderr.txt");
    command
        .stdin(Stdio::null())
        .stdout(File::create(&stdout).unwrap())
        .stderr(File::create(&stderr).unwrap());
    let mut child = Reap(command.spawn().expect("launch official djxl"));
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let start = Instant::now();
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(
                status.success(),
                "djxl {status}:\n{}\n{}",
                fs::read_to_string(&stdout).unwrap(),
                fs::read_to_string(&stderr).unwrap()
            );
            return;
        }
        assert!(
            start.elapsed() < Duration::from_secs(90),
            "djxl timed out (guard kills/reaps child)"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn compare_png(path: &Path, image: ImageView<'_>, label: &str) {
    assert!(matches!(image.bit_depth, 8 | 16));
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path).unwrap()));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    let expected_color = match image.channels {
        1 => png::ColorType::Grayscale,
        2 => png::ColorType::GrayscaleAlpha,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        _ => unreachable!(),
    };
    assert_eq!(
        (reader.info().width as usize, reader.info().height as usize),
        (image.width, image.height),
        "{label}"
    );
    let mut bytes = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut bytes).unwrap();
    assert_eq!(info.color_type, expected_color, "{label}");
    let storage_bits = if image.bit_depth <= 8 { 8 } else { 16 };
    assert_eq!(info.bit_depth as u8, storage_bits, "{label}");
    let samples: Vec<_> = if storage_bits == 8 {
        bytes[..info.buffer_size()]
            .iter()
            .map(|&v| u16::from(v))
            .collect()
    } else {
        bytes[..info.buffer_size()]
            .chunks_exact(2)
            .map(|v| u16::from_be_bytes([v[0], v[1]]))
            .collect()
    };
    let mut i = 0;
    for y in 0..image.height {
        for x in 0..image.width {
            for c in 0..usize::from(image.channels) {
                let expected = image.sample(x, y, c);
                assert_eq!(
                    samples[i], expected,
                    "{label}: ({x},{y}) channel {c}; hidden RGB is compared too"
                );
                i += 1;
            }
        }
    }
    assert_eq!(samples.len(), i, "{label}");
}

// Only the fixed dictionary emitted by b866b3d extras/enc/npy.cc::WriteNPYHeader,
// not arbitrary Python syntax. Validate shape and the entire payload length.
fn npy_samples(bytes: &[u8], shape: [usize; 4]) -> std::result::Result<Vec<f32>, &'static str> {
    if bytes.len() < 10 || &bytes[..8] != b"\x93NUMPY\x01\x00" {
        return Err("expected NPY version 1.0");
    }
    let end = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let header = bytes.get(10..end).ok_or("truncated NPY header")?;
    if !header.is_ascii() || !header.ends_with(b"\n") {
        return Err("expected newline-terminated ASCII NPY header");
    }
    let header = std::str::from_utf8(header).map_err(|_| "invalid NPY header")?;
    let [frames, height, width, channels] = shape;
    let expected = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({frames}, {height}, {width}, {channels}), }}"
    );
    if header.trim_end() != expected {
        return Err("unexpected NPY dtype, order or shape");
    }
    let count = shape
        .into_iter()
        .try_fold(1usize, |n, d| n.checked_mul(d))
        .ok_or("NPY dimensions overflow")?;
    let payload = &bytes[end..];
    if count == 0 || count.checked_mul(4) != Some(payload.len()) {
        return Err("NPY payload length does not match shape");
    }
    Ok(payload
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

fn recover_integer(sample: f32, depth: u8) -> u16 {
    assert!((1..=16).contains(&depth));
    assert!(
        sample.is_finite() && (0.0..=1.0).contains(&sample),
        "invalid normalized integer {sample:?}"
    );
    (f64::from(sample) * f64::from((1u32 << depth) - 1)).round() as u16
}

fn compare_npy(path: &Path, image: ImageView<'_>, label: &str) {
    let bytes = fs::read(path).unwrap();
    let samples = npy_samples(
        &bytes,
        [1, image.height, image.width, usize::from(image.channels)],
    )
    .unwrap_or_else(|error| panic!("{label}: {error}"));
    let mut i = 0;
    for y in 0..image.height {
        for x in 0..image.width {
            for c in 0..usize::from(image.channels) {
                let expected = image.sample(x, y, c);
                let actual = recover_integer(samples[i], image.bit_depth);
                assert_eq!(
                    actual,
                    expected,
                    "{label}: ({x},{y}) channel {c}, float bits={:08x}; hidden RGB is compared too",
                    samples[i].to_bits()
                );
                i += 1;
            }
        }
    }
    assert_eq!(samples.len(), i, "{label}");
}

#[test]
fn source_uint8_dither_explains_seven_bit_png_counterexample() {
    // dec_modular.cc::SingleFromSingle uses an f32 reciprocal and multiply.
    let normalized = 82.0f32 * (1.0f64 / 127.0) as f32;
    assert_eq!(normalized.to_bits(), 0x3f25_4a95);
    let scaled = normalized * 255.0;
    assert_eq!(scaled.round(), 165.0);
    // stage_write.cc::MakeUnsigned<uint8_t>: x=1,y=0,c=1 selects
    // kDither[((0 + 13) % 32) * 48 + (1 + 23) % 32] == -0.44200.
    assert_eq!((scaled + (-0.44200f32)).round(), 164.0);
    assert_eq!(recover_integer(normalized, 7), 82);
}

#[test]
fn normalized_f32_preserves_every_integer_through_sixteen_bits() {
    for depth in 1..=16 {
        let max = (1u32 << depth) - 1;
        let factor = (1.0f64 / f64::from(max)) as f32;
        let mut previous = None;
        for original in 0..=max {
            let source = original as f32 * factor;
            let accurate = (f64::from(original) / f64::from(max)) as f32;
            assert_eq!(recover_integer(source, depth), original as u16);
            assert_eq!(recover_integer(accurate, depth), original as u16);
            if let Some(previous) = previous {
                assert!(source > previous);
            }
            previous = Some(source);
        }
    }
}

#[test]
fn official_npy_shape_type_and_payload_are_checked() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 2, 1), }\n";
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.bytes());
    bytes.extend(0.0f32.to_le_bytes());
    bytes.extend(1.0f32.to_le_bytes());
    assert_eq!(npy_samples(&bytes, [1, 1, 2, 1]).unwrap(), [0.0, 1.0]);
    assert!(npy_samples(&bytes, [1, 1, 1, 2]).is_err());
    for end in [0, 8, 9, bytes.len() - 1] {
        assert!(npy_samples(&bytes[..end], [1, 1, 2, 1]).is_err());
    }
    bytes.push(0);
    assert!(npy_samples(&bytes, [1, 1, 2, 1]).is_err());
    bytes.pop();
    let dtype = bytes.windows(3).position(|s| s == b"<f4").unwrap();
    bytes[dtype + 2] = b'8';
    assert!(npy_samples(&bytes, [1, 1, 2, 1]).is_err());
}

#[test]
#[ignore = "requires RS_JXL_DJXL OS path list; official independent pixel oracle"]
fn official_scalar_and_native_fjxl_profiles() {
    let paths = env::var_os("RS_JXL_DJXL").expect("set RS_JXL_DJXL to official djxl OS path list");
    let decoders: Vec<_> = env::split_paths(&paths)
        .map(|p| {
            assert!(p.is_file());
            p.canonicalize().unwrap()
        })
        .collect();
    assert!(!decoders.is_empty());
    let scratch = Scratch::new();
    for decoder in &decoders {
        run(Command::new(decoder).arg("--version"), &scratch.0);
        let banner = format!(
            "{}{}",
            fs::read_to_string(scratch.0.join("stdout.txt")).unwrap(),
            fs::read_to_string(scratch.0.join("stderr.txt")).unwrap()
        );
        assert!(banner.contains("djxl v"), "not an official djxl: {banner}");
    }
    // Four storage/depth profiles, narrow tails, multiple AC/DC groups, and
    // palette indices switching from global channel 1 to local channel 0.
    let cases = [
        (1, 1, 1, 8, false, 0, true),
        (7, 3, 2, 8, false, 1, true),
        (17, 9, 3, 8, false, 2, true),
        (33, 7, 4, 8, true, 2, true),
        (257, 3, 4, 8, true, 2, true),
        (17, 9, 1, 8, true, 2, true),
        (17, 5, 2, 8, true, 2, true),
        (257, 17, 3, 8, false, 2, false),
        (17, 3, 3, 1, false, 0, true),
        (9, 3, 4, 7, false, 2, true),
        (17, 5, 3, 9, false, 2, true),
        (17, 5, 2, 13, false, 2, true),
        (17, 5, 4, 14, false, 2, true),
        (17, 5, 3, 15, false, 2, true),
        (257, 3, 4, 16, false, 2, true),
        (17, 5, 1, 16, false, 2, true),
        (2049, 1, 3, 8, false, 0, true),
        (17, 5, 4, 8, false, 127, true),
        (257, 17, 3, 8, false, 64, false),
    ];
    for (case, &(width, height, channels, depth, palette, effort, oneshot)) in
        cases.iter().enumerate()
    {
        let max = (1u32 << depth) - 1;
        let bytes_per_sample = if depth <= 8 { 1 } else { 2 };
        let stride = width * usize::from(channels) * bytes_per_sample + 3;
        let mut input = vec![0x79; stride * height];
        let colors = [
            [0, 0, 0, 0],
            [13, 227, 7, 255],
            [199, 31, 241, 0],
            [63, 171, 29, 50],
        ];
        for y in 0..height {
            for x in 0..width {
                let color = &colors[(x / 3 + y / 2) % colors.len()];
                for (c, &palette_value) in color.iter().take(usize::from(channels)).enumerate() {
                    let n = if palette {
                        palette_value
                    } else if c == 3 && x % 3 == 0 {
                        0
                    } else {
                        ((x * 8191 + y * 971 + c * 211 + x * y) as u32) & max
                    };
                    let i = y * stride + (x * usize::from(channels) + c) * bytes_per_sample;
                    if bytes_per_sample == 1 {
                        input[i] = n as u8;
                    } else {
                        input[i..i + 2].copy_from_slice(&(n as u16).to_be_bytes());
                    }
                }
            }
        }
        let image = ImageView {
            pixels: &input,
            width,
            height,
            row_stride: stride,
            channels,
            bit_depth: depth,
            big_endian: true,
        };
        for backend in [Backend::Scalar, Backend::Auto] {
            let options = Options {
                effort,
                oneshot,
                backend,
                ..Options::default()
            };
            let label = format!(
                "case={case} {width}x{height} c={channels} depth={depth} palette={palette} effort={effort} oneshot={oneshot} {backend:?}"
            );
            let bytes = encode(image, &options).unwrap_or_else(|e| panic!("{label}: {e}"));
            let file = scratch.0.join("input.jxl");
            fs::write(&file, bytes).unwrap();
            for decoder in &decoders {
                // UINT8 PNG output is dithered at non-storage bit depths.
                // NPY bypasses integer rendering; recover original integer bits.
                let png = matches!(depth, 8 | 16);
                let output = scratch
                    .0
                    .join(if png { "output.png" } else { "output.npy" });
                assert!(!output.exists());
                let mut command = Command::new(decoder);
                command.arg(&file).arg(&output);
                if png {
                    command.arg(format!("--bits_per_sample={depth}"));
                }
                run(command.arg("--num_threads=1").arg("--quiet"), &scratch.0);
                if png {
                    compare_png(&output, image, &label);
                } else {
                    compare_npy(&output, image, &label);
                }
                fs::remove_file(output).unwrap();
            }
            eprintln!("FJXL oracle {label}");
        }
    }
}

#[test]
#[ignore = "requires RS_JXL_DJXL OS path list; bounded region/streaming pixel oracle"]
fn official_region_incremental_and_seek_output() {
    use super::regions::tests::{Generated, MemoryOutput};

    let paths = env::var_os("RS_JXL_DJXL").expect("set RS_JXL_DJXL to official djxl OS path list");
    let decoders: Vec<_> = env::split_paths(&paths)
        .map(|p| p.canonicalize().unwrap())
        .collect();
    assert!(!decoders.is_empty());
    let scratch = Scratch::new();
    for decoder in &decoders {
        run(Command::new(decoder).arg("--version"), &scratch.0);
        let banner = format!(
            "{}{}",
            fs::read_to_string(scratch.0.join("stdout.txt")).unwrap(),
            fs::read_to_string(scratch.0.join("stderr.txt")).unwrap()
        );
        assert!(banner.contains("djxl v"), "not an official djxl: {banner}");
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    // One group, palette + hidden RGB, vertical/horizontal partial groups,
    // narrow uint14, uint16, multiple DC groups and a second sixteen-group batch.
    let cases = [
        (17, 3, 2, 7, false),
        (257, 5, 4, 8, true),
        (257, 257, 3, 14, false),
        (4097, 3, 4, 16, false),
    ];
    for (case, (width, height, channels, depth, palette)) in cases.into_iter().enumerate() {
        let source = Generated::new(width, height, channels, depth, palette);
        let (pixels, stride) = source.expected_pixels();
        let image = source.image(&pixels, stride);
        for backend in [Backend::Scalar, Backend::Auto] {
            let options = Options {
                effort: if case == 3 { 64 } else { 2 },
                oneshot: palette,
                backend,
                ..Options::default()
            };
            for streaming in [false, true] {
                let prepared = prepare_frame(&source, &options).unwrap();
                let bytes = if streaming {
                    let mut output = MemoryOutput::new(97);
                    // Parent can write metadata first and stream frame-only at
                    // a nonzero absolute offset, leaving container finalization open.
                    let mut metadata = BitWriter::new(128);
                    header::image(&mut metadata, image).unwrap();
                    output.bytes.extend_from_slice(metadata.bytes().unwrap());
                    output.seek(output.bytes.len() as u64).unwrap();
                    let count = prepared
                        .process_to_with_pool(&mut output, false, true, &pool)
                        .unwrap();
                    assert_eq!(count + metadata.bytes().unwrap().len(), output.bytes.len());
                    output.bytes
                } else {
                    let mut frame = prepared
                        .process_with_pool(&pool)
                        .unwrap()
                        .prepare_header(true, true)
                        .unwrap();
                    let mut bytes = Vec::new();
                    let mut chunk = [0; 31];
                    while frame.remaining_size() != 0 {
                        let count = frame.write_output(&mut chunk).unwrap();
                        bytes.extend_from_slice(&chunk[..count]);
                    }
                    assert_eq!(frame.write_output(&mut []).unwrap(), 0);
                    bytes
                };
                let label = format!(
                    "region case={case} {width}x{height} c={channels} depth={depth} {backend:?} seek={streaming}"
                );
                let file = scratch.0.join("input.jxl");
                fs::write(&file, bytes).unwrap();
                for decoder in &decoders {
                    let png = matches!(depth, 8 | 16);
                    let output = scratch
                        .0
                        .join(if png { "output.png" } else { "output.npy" });
                    assert!(!output.exists());
                    let mut command = Command::new(decoder);
                    command.arg(&file).arg(&output);
                    if png {
                        command.arg(format!("--bits_per_sample={depth}"));
                    }
                    run(command.arg("--num_threads=1").arg("--quiet"), &scratch.0);
                    if png {
                        compare_png(&output, image, &label);
                    } else {
                        compare_npy(&output, image, &label);
                    }
                    fs::remove_file(output).unwrap();
                }
                eprintln!("FJXL oracle {label}");
            }
        }
    }
}
