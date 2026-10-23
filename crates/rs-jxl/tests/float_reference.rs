//! Official djxl NPY interoperability, without Python or a Rust JXL decoder.
//!
//! RS_JXL_DJXL is the same OS path list as tests/reference.rs. For example:
//! cargo test -p rs-jxl --test float_reference finite_ -- --ignored --nocapture
//! cargo test -p rs-jxl --test float_reference u17_ -- --ignored --nocapture
//! cargo test -p rs-jxl --test float_reference u31_ -- --ignored --nocapture
//! cargo test -p rs-jxl --test float_reference experimental_ -- --ignored --nocapture
//! cargo test -p rs-jxl --test float_reference active_ -- --ignored --nocapture
//! cargo test -p rs-jxl --test float_reference global_layout_ -- --ignored --nocapture
//!
//! The experimental NaN/Inf tests are separate from the finite-value corpus.
//! Explicit invocation requires success and exact words; decoder failures or
//! NaN canonicalization are failures, not skips or accepted alternatives.
//! NPY stores f32: the U31 test cannot establish full integer-bit preservation.
//! Wide integer extras under a <23-bit main image have the same limitation
//! because official dec_modular.cc uses a rounded f32 scale factor there.
//! Active cases exercise learned constant offsets. Explicit custom trees also
//! force non-unit multipliers, with every wide residual checked for exact
//! divisibility before official decoding, independently of MA training.

use std::env;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::entropy::stream::Coding;
use rs_jxl::metadata::{BitDepth, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::layout::Passes;
use rs_jxl::modular::palette::PaletteOptions;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::modular::sample::{SamplingPolicy, SourcePolicy};
use rs_jxl::modular::tree::{Node, Tree};
use rs_jxl::modular::{GroupwiseDecision, groupwise_decision};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode};

const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Self {
        let root = env::temp_dir()
            .canonicalize()
            .expect("resolve temp directory");
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        for _ in 0..100 {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "rs-jxl-float-reference-{}-{timestamp}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        }
        panic!("could not create a unique float-reference directory");
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

fn run_checked(command: &mut Command, scratch: &Path, context: &str) {
    let stdout = scratch.join("stdout.log");
    let stderr = scratch.join("stderr.log");
    // File redirection cannot deadlock on a full diagnostic pipe.
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    let result = wait_with_timeout(command);
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let diagnostics = format!(
        "stdout:\n{}\nstderr:\n{}",
        fs::read_to_string(stdout).unwrap_or_default(),
        fs::read_to_string(stderr).unwrap_or_default()
    );
    match result {
        Ok(Some(status)) if status.success() => {}
        Ok(Some(status)) => panic!("{context}: djxl exited {status}\n{diagnostics}"),
        Ok(None) => {
            panic!("{context}: djxl exceeded {COMMAND_TIMEOUT:?}; killed and reaped\n{diagnostics}")
        }
        Err(error) => panic!("{context}: run djxl: {error}\n{diagnostics}"),
    }
}

fn decoder_paths() -> Vec<PathBuf> {
    let value = env::var_os("RS_JXL_DJXL").expect(
        "explicit reference test requires RS_JXL_DJXL: an OS path list of official djxl executables",
    );
    let paths: Vec<_> = env::split_paths(&value).collect();
    assert!(!paths.is_empty(), "RS_JXL_DJXL contains no decoder paths");
    paths
        .into_iter()
        .map(|path| {
            assert!(
                !path.as_os_str().is_empty() && path.is_file(),
                "RS_JXL_DJXL entry is empty or not a file: {}",
                path.display()
            );
            path.canonicalize()
                .unwrap_or_else(|error| panic!("resolve {}: {error}", path.display()))
        })
        .collect()
}

#[derive(Debug)]
struct Npy {
    shape: [usize; 4],
    words: Vec<u32>,
}

/// Parses the small fixed dictionary written by lib/extras/enc/npy.cc, not
/// arbitrary Python literals. Official output may omit NumPy's usual padding.
fn parse_npy(bytes: &[u8]) -> Result<Npy, String> {
    if bytes.len() < 10 || &bytes[..8] != b"\x93NUMPY\x01\x00" {
        return Err("expected NPY version 1.0".into());
    }
    let length = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let header_end = 10 + length;
    let header = bytes.get(10..header_end).ok_or("truncated NPY header")?;
    if !header.is_ascii() || !header.ends_with(b"\n") {
        return Err("expected newline-terminated ASCII NPY header".into());
    }
    let header = std::str::from_utf8(header)
        .map_err(|_| "invalid NPY header")?
        .trim_end();
    let dimensions = header
        .strip_prefix("{'descr': '<f4', 'fortran_order': False, 'shape': (")
        .and_then(|h| h.strip_suffix("), }"))
        .ok_or("expected official C-order <f4 NPY dictionary")?;
    let dimensions: Vec<_> = dimensions.split(',').map(str::trim).collect();
    if dimensions.len() != 4 {
        return Err("expected four NPY dimensions".into());
    }
    let mut shape = [0usize; 4];
    let mut elements = 1usize;
    for (destination, dimension) in shape.iter_mut().zip(dimensions) {
        if dimension.is_empty() || !dimension.bytes().all(|b| b.is_ascii_digit()) {
            return Err("invalid NPY dimension".into());
        }
        *destination = dimension.parse().map_err(|_| "NPY dimension overflow")?;
        if *destination == 0 {
            return Err("empty NPY dimension".into());
        }
        elements = elements
            .checked_mul(*destination)
            .ok_or("NPY element count overflow")?;
    }
    let payload = &bytes[header_end..];
    if elements.checked_mul(4) != Some(payload.len()) {
        return Err("NPY payload length does not match shape".into());
    }
    let words = payload
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    Ok(Npy { shape, words })
}

// Exact binary16-to-binary32 widening, independent of the encoder. Keeping
// words as integers also avoids quieting a signaling NaN in the test itself.
fn half_to_f32_bits(word: u16) -> u32 {
    let sign = u32::from(word & 0x8000) << 16;
    let exponent = u32::from((word >> 10) & 31);
    let fraction = u32::from(word & 1023);
    match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = 31 - fraction.leading_zeros();
            sign | ((leading + 103) << 23) | ((fraction - (1 << leading)) << (23 - leading))
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    }
}

fn direct_options(coding: Coding) -> EncodeOptions {
    EncodeOptions {
        modular: ModularOptions {
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            rct: None,
            lz77: false,
            entropy: coding,
            tree_samples: 1,
            max_tree_leaves: 1,
            ..ModularOptions::default()
        },
        ..EncodeOptions::default()
    }
}

fn assert_samples(actual: &Npy, image: &Image, context: &str) {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let channels = image.channel_count();
    assert_eq!(
        actual.shape,
        [1, height, width, channels],
        "{context}: NPY shape"
    );
    for pixel in 0..width * height {
        for channel in 0..channels {
            let (data, index) = if channel < image.layout().channels() {
                (image.pixels(), pixel * image.layout().channels() + channel)
            } else {
                (
                    &image.extra_planes()[channel - image.layout().channels()],
                    pixel,
                )
            };
            let word = actual.words[pixel * channels + channel];
            let location = format!(
                "{context}: x={} y={} channel={channel}",
                pixel % width,
                pixel / width
            );
            let expected = match data {
                PixelData::F16(values) => half_to_f32_bits(values[index]),
                PixelData::F32(values) => values[index].to_bits(),
                PixelData::U32(values) => {
                    let bits = image.channel_depth(channel).bits_per_sample;
                    let maximum = (1u32 << bits) - 1;
                    let decoded = f32::from_bits(word);
                    assert!(
                        decoded.is_finite() && (0.0..=1.0).contains(&decoded),
                        "{location}: invalid normalized integer {decoded}"
                    );
                    // Source dec_modular.cc selects SIMD float multiplication
                    // for ALL channels when the MAIN bit depth is below 23.
                    // Wide extras then have a weaker oracle than wide mains.
                    let float_factor = image.bits() < 23;
                    if bits <= 24 && !(float_factor && bits >= 23) {
                        let recovered = (f64::from(decoded) * f64::from(maximum)).round();
                        assert_eq!(
                            recovered,
                            f64::from(values[index]),
                            "{location}: U{bits}, NPY word={word:08x}"
                        );
                        continue;
                    }
                    assert!(bits == 31 || (float_factor && (23..=24).contains(&bits)));
                    // Exact SOURCE projection, not an epsilon comparison or a
                    // claim that the original integer is uniquely recoverable.
                    integer_projection_bits(values[index], bits, float_factor)
                }
                _ => panic!("unexpected sample storage in float oracle fixture"),
            };
            assert_eq!(
                word, expected,
                "{location}: expected word={expected:08x}, decoded word={word:08x}"
            );
        }
    }
}

fn integer_projection_bits(value: u32, bits: u8, float_factor: bool) -> u32 {
    let factor = 1.0 / f64::from((1u32 << bits) - 1);
    if float_factor {
        (value as f32 * factor as f32).to_bits()
    } else {
        ((f64::from(value) * factor) as f32).to_bits()
    }
}

fn check_image(image: &Image, label: &str, decoders: &[PathBuf], scratch: &ScratchDir) {
    for coding in [Coding::Prefix, Coding::Ans] {
        check_image_with_options(
            image,
            &direct_options(coding),
            &format!("{label}/{coding:?}"),
            decoders,
            scratch,
        );
    }
}

fn check_image_with_options(
    image: &Image,
    options: &EncodeOptions,
    label: &str,
    decoders: &[PathBuf],
    scratch: &ScratchDir,
) {
    for channel in 0..image.channel_count() {
        let depth = image.channel_depth(channel);
        if !depth.is_float()
            && (depth.bits_per_sample > 24 || (image.bits() < 23 && depth.bits_per_sample >= 23))
        {
            eprintln!(
                "{label}: channel {channel} U{} checks ONLY exact f32 projection (main depth {}), not complete integer precision",
                depth.bits_per_sample,
                image.bits()
            );
        }
    }
    let data = encode(image, options).unwrap_or_else(|error| panic!("{label}: encode: {error}"));
    check_encoded_image(image, &data, label, decoders, scratch);
}

fn check_encoded_image(
    image: &Image,
    data: &[u8],
    label: &str,
    decoders: &[PathBuf],
    scratch: &ScratchDir,
) {
    let input = scratch.0.join("input.jxl");
    fs::write(&input, data).unwrap();
    for decoder in decoders {
        let output = scratch.0.join("output.npy");
        assert!(!output.exists(), "previous oracle output was not removed");
        let context = format!("{label}/{}", decoder.display());
        // NPY's accepted format is explicitly float32. Do not request an
        // output color conversion, alpha compositing, or integer PNG depth.
        run_checked(
            Command::new(decoder)
                .arg(&input)
                .arg(&output)
                .arg("--num_threads=1")
                .arg("--quiet"),
            &scratch.0,
            &context,
        );
        let bytes =
            fs::read(&output).unwrap_or_else(|error| panic!("{context}: read NPY: {error}"));
        let actual = parse_npy(&bytes).unwrap_or_else(|error| panic!("{context}: {error}"));
        assert_samples(&actual, image, &context);
        fs::remove_file(output).unwrap();
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn finite_half_fast_lossless_preserves_original_words() {
    check_fast_half_words(false);
}

#[test]
#[ignore = "requires official djxl; FJXL signed zero, infinity and NaN payload words"]
fn experimental_half_fast_lossless_preserves_original_words() {
    check_fast_half_words(true);
}

fn check_fast_half_words(special: bool) {
    use rs_jxl::scheduler::native::encode_native_candidate;
    use rs_jxl::scheduler::params::CandidateParams;
    use rs_jxl::scheduler::{NativeBackendOptions, NativeImplementation};

    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for layout in [
        PixelLayout::Gray,
        PixelLayout::GrayAlpha,
        PixelLayout::Rgb,
        PixelLayout::Rgba,
    ] {
        let count = 263 * 9 * layout.channels();
        let pixels = if special {
            let words = [
                0, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0xfe55, 0x7c01, 0xfc01, 0x7fff, 0xffff,
            ];
            PixelData::F16((0..count).map(|i| words[i % words.len()]).collect())
        } else {
            finite_pixels(true, count, 3)
        };
        let image = Image::new(263, 9, layout, pixels).unwrap();
        for threads in [1, 4] {
            let params = CandidateParams {
                speed_tier: 9,
                ..Default::default()
            };
            let options = EncodeOptions {
                threads,
                ..Default::default()
            };
            let encoded =
                encode_native_candidate(&params, &image, &options, NativeBackendOptions::default())
                    .unwrap();
            assert_eq!(
                encoded.artifact.implementation,
                NativeImplementation::FastLossless
            );
            check_encoded_image(
                &image,
                &encoded.bytes,
                &format!("F16/FJXL/{layout:?}/special={special}/threads={threads}"),
                &decoders,
                &scratch,
            );
        }
    }
}

#[test]
#[ignore = "requires official djxl; seekable region input preserves F16/F32 words"]
fn finite_region_input_preserves_original_words() {
    use rs_jxl::{region_input::ImageRegionSource, streaming::encode_regions_seekable_to};
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for half in [false, true] {
        for layout in [PixelLayout::GrayAlpha, PixelLayout::Rgba] {
            for (width, height) in [(1025, 3), (3, 1025)] {
                let image = Image::new(
                    width,
                    height,
                    layout,
                    finite_pixels(
                        half,
                        width as usize * height as usize * layout.channels(),
                        7,
                    ),
                )
                .unwrap();
                let mut options = EncodeOptions {
                    threads: 4,
                    group_size_shift: 0,
                    ..Default::default()
                };
                options.modular.palette_policy = Some(PaletteOptions {
                    groupwise: true,
                    palette_colors: 0,
                    channel_colors_percent: 0.0,
                    channel_colors_pre_transform_percent: 0.0,
                    ..Default::default()
                });
                let mut source = ImageRegionSource::new(&image).unwrap();
                let mut output = std::io::Cursor::new(Vec::new());
                encode_regions_seekable_to(&mut source, &options, &mut output).unwrap();
                check_encoded_image(
                    &image,
                    output.get_ref(),
                    &format!("region/{width}x{height}/{layout:?}/half={half}"),
                    &decoders,
                    &scratch,
                );
            }
        }
    }
}

const HALF_FINITE: &[u16] = &[
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3555, 0xb555, 0x3800, 0xb800,
    0x3bff, 0x3c00, 0x3c01, 0xbc00, 0x4000, 0xc000, 0x5800, 0xd800, 0x7bff, 0xfbff,
];
const FLOAT_FINITE: &[u32] = &[
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x0080_0000,
    0x8080_0000,
    0x3dcccccd,
    0xbdcccccd,
    0x3eaaaaab,
    0xbeaaaaab,
    0x3f00_0000,
    0xbf00_0000,
    0x3f7f_ffff,
    0x3f80_0000,
    0x3f80_0001,
    0xbf80_0000,
    0x4000_0000,
    0xc000_0000,
    0x4b00_0001,
    0xcb00_0001,
    0x7f00_0000,
    0xff00_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
];

fn finite_pixels(half: bool, count: usize, seed: usize) -> PixelData {
    if half {
        PixelData::F16(
            (0..count)
                .map(|i| HALF_FINITE[(i * 7 + i / HALF_FINITE.len() + seed) % HALF_FINITE.len()])
                .collect(),
        )
    } else {
        PixelData::F32(
            (0..count)
                .map(|i| {
                    f32::from_bits(
                        FLOAT_FINITE[(i * 7 + i / FLOAT_FINITE.len() + seed) % FLOAT_FINITE.len()],
                    )
                })
                .collect(),
        )
    }
}

fn integer_pixels(bits: u8, count: usize, seed: usize) -> PixelData {
    let maximum = (1u32 << bits) - 1;
    let midpoint = 1u32 << (bits - 1);
    let edges = [
        0,
        maximum,
        1,
        maximum - 1,
        midpoint - 1,
        midpoint,
        midpoint + 1,
        65535,
        65536,
        65537,
        (1 << 23) - 1,
        1 << 23,
        (1 << 24) - 1,
        1 << 24,
    ];
    PixelData::U32(
        (0..count)
            .map(|i| {
                let position = (i + i / 16 * 3 + seed) % 16;
                if position < edges.len() {
                    edges[position].min(maximum)
                } else {
                    (i as u32)
                        .wrapping_mul(0x9e37_79b9)
                        .wrapping_add(seed as u32)
                        & maximum
                }
            })
            .collect(),
    )
}

fn with_extras(image: Image, planes: Vec<(PixelData, BitDepth)>) -> Image {
    let mut metadata = image.metadata().clone();
    let mut data = Vec::new();
    for (i, (pixels, depth)) in planes.into_iter().enumerate() {
        let kind = [
            ExtraChannelType::Depth,
            ExtraChannelType::SelectionMask,
            ExtraChannelType::Thermal,
        ][i];
        let mut extra = ExtraChannelInfo::new(kind, 8).unwrap();
        extra.bit_depth = depth;
        extra.name = format!("word-plane-{i}");
        metadata.extra_channels.push(extra);
        data.push(pixels);
    }
    image.with_metadata(metadata, data).unwrap()
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL"]
fn finite_f16_f32_color_and_alpha_word_wire() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for half in [true, false] {
        for layout in [
            PixelLayout::Gray,
            PixelLayout::GrayAlpha,
            PixelLayout::Rgb,
            PixelLayout::Rgba,
        ] {
            for (width, height) in [(17, 5), (259, 3)] {
                let count = width as usize * height as usize * layout.channels();
                let image =
                    Image::new(width, height, layout, finite_pixels(half, count, 0)).unwrap();
                check_image(
                    &image,
                    &format!(
                        "finite/{}/{layout:?}/{width}x{height}",
                        if half { "F16" } else { "F32" }
                    ),
                    &decoders,
                    &scratch,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL"]
fn finite_mixed_float_extra_channels_word_wire() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for half in [true, false] {
        for layout in [PixelLayout::GrayAlpha, PixelLayout::Rgba] {
            let (width, height) = (259, 3);
            let count = width as usize * height as usize;
            let base = Image::new(
                width,
                height,
                layout,
                finite_pixels(half, count * layout.channels(), 3),
            )
            .unwrap();
            let image = with_extras(
                base,
                vec![
                    (
                        finite_pixels(true, count, 5),
                        BitDepth::floating(16, 5).unwrap(),
                    ),
                    (
                        finite_pixels(false, count, 11),
                        BitDepth::floating(32, 8).unwrap(),
                    ),
                    (integer_pixels(24, count, 7), BitDepth::new(24).unwrap()),
                ],
            );
            check_image(
                &image,
                &format!("finite/mixed-extras/half={half}/{layout:?}"),
                &decoders,
                &scratch,
            );
        }
    }
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL"]
fn u17_through_u24_integer_recovery_from_official_npy() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for bits in 17..=24 {
        for layout in [
            PixelLayout::Gray,
            PixelLayout::GrayAlpha,
            PixelLayout::Rgb,
            PixelLayout::Rgba,
        ] {
            let count = 17 * 5;
            let image = Image::new(
                17,
                5,
                layout,
                integer_pixels(bits, count * layout.channels(), 0),
            )
            .unwrap();
            let mut metadata = image.metadata().clone();
            metadata.bit_depth = BitDepth::new(bits).unwrap();
            for alpha in &mut metadata.extra_channels {
                alpha.bit_depth = metadata.bit_depth;
            }
            let image = image.with_metadata(metadata, vec![]).unwrap();
            let image = with_extras(
                image,
                vec![(integer_pixels(bits, count, 5), BitDepth::new(bits).unwrap())],
            );
            check_image(
                &image,
                &format!("U{bits}/{layout:?}/integer-recovery"),
                &decoders,
                &scratch,
            );
        }
    }
}

#[test]
#[ignore = "requires official djxl; NPY verifies only the f32 projection, NOT full U31 precision"]
fn u31_official_float_projection_not_full_integer_precision() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let maximum = (1u32 << 31) - 1;
    let project = |v: u32| (f64::from(v) / f64::from(maximum)) as f32;
    assert_eq!(project(maximum - 1).to_bits(), project(maximum).to_bits());
    eprintln!(
        "U31 oracle limitation: official NPY is f32; max-1 and max collide. This test does not prove all integer bits survived."
    );
    for layout in [
        PixelLayout::Gray,
        PixelLayout::GrayAlpha,
        PixelLayout::Rgb,
        PixelLayout::Rgba,
    ] {
        let count = 259 * 3;
        let base = Image::new(
            259,
            3,
            layout,
            integer_pixels(31, count * layout.channels(), 0),
        )
        .unwrap();
        let image = with_extras(
            base,
            vec![(integer_pixels(31, count, 3), BitDepth::new(31).unwrap())],
        );
        check_image(
            &image,
            &format!("U31/{layout:?}/float-projection-only"),
            &decoders,
            &scratch,
        );
    }
}

// Construct exactly representable custom-format numbers mathematically. Do
// not use the encoder's float_to_int/int_to_float to manufacture its own oracle.
fn custom_float_values(depth: BitDepth) -> Vec<f32> {
    let exponent_bits = u32::from(depth.exponent_bits_per_sample);
    let mantissa_bits = u32::from(depth.bits_per_sample) - exponent_bits - 1;
    let bias = (1u32 << (exponent_bits - 1)) - 1;
    let exponent_max = (1u32 << exponent_bits) - 1;
    let mantissa_max = (1u32 << mantissa_bits) - 1;
    let mut magnitudes = vec![0u32];
    // The source F32->custom mapping inserts a hidden leading bit for exp8
    // subnormals. Its strict inverse guard rejects those, even if the numeric
    // layout could represent them. They have a separate rejection test below.
    if exponent_bits < 8 {
        for mantissa in [1, 2, 1 << (mantissa_bits - 1), mantissa_max] {
            let value = f64::from(mantissa) * 2.0f64.powi(1 - bias as i32 - mantissa_bits as i32);
            magnitudes.push((value as f32).to_bits());
        }
    }
    for exponent in [1, bias - 1, bias, bias + 1, exponent_max - 1] {
        for mantissa in [0, 1, 1 << (mantissa_bits - 1), mantissa_max] {
            let value = (1.0 + f64::from(mantissa) / f64::from(1u32 << mantissa_bits))
                * 2.0f64.powi(exponent as i32 - bias as i32);
            assert!(value.is_finite() && (value as f32).is_finite());
            magnitudes.push((value as f32).to_bits());
        }
    }
    magnitudes
        .into_iter()
        .flat_map(|bits| [f32::from_bits(bits), f32::from_bits(bits | 0x8000_0000)])
        .collect()
}

fn custom_pixels(depth: BitDepth, count: usize, seed: usize) -> PixelData {
    let values = custom_float_values(depth);
    PixelData::F32(
        (0..count)
            .map(|i| values[(i + i / values.len() + seed) % values.len()])
            .collect(),
    )
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL"]
fn finite_custom_float_metadata_color_alpha_and_extras() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let formats = [(24, 8), (16, 5), (10, 5), (8, 3)];
    for (bits, exponent) in formats {
        let depth = BitDepth::floating(bits, exponent).unwrap();
        for layout in [
            PixelLayout::Gray,
            PixelLayout::GrayAlpha,
            PixelLayout::Rgb,
            PixelLayout::Rgba,
        ] {
            let count = 17 * 5;
            let base = Image::new(
                17,
                5,
                layout,
                custom_pixels(depth, count * layout.channels(), 0),
            )
            .unwrap();
            let mut metadata = base.metadata().clone();
            metadata.bit_depth = depth;
            for alpha in &mut metadata.extra_channels {
                alpha.bit_depth = depth;
            }
            let base = base.with_metadata(metadata, vec![]).unwrap();
            let extras = [(24, 8), (16, 5), (10, 5)]
                .into_iter()
                .enumerate()
                .map(|(i, (bits, exponent))| {
                    let depth = BitDepth::floating(bits, exponent).unwrap();
                    (custom_pixels(depth, count, i * 3 + 1), depth)
                })
                .collect();
            let image = with_extras(base, extras);
            check_image(
                &image,
                &format!("finite/custom-{bits}-{exponent}/{layout:?}"),
                &decoders,
                &scratch,
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum LayoutFloat {
    Half,
    Single,
    Custom(u8, u8),
}

impl LayoutFloat {
    fn depth(self) -> BitDepth {
        let (bits, exponent) = match self {
            Self::Half => (16, 5),
            Self::Single => (32, 8),
            Self::Custom(bits, exponent) => (bits, exponent),
        };
        BitDepth::floating(bits, exponent).unwrap()
    }

    fn pixels(self, count: usize, seed: usize) -> PixelData {
        match self {
            Self::Half => finite_pixels(true, count, seed),
            Self::Single => finite_pixels(false, count, seed),
            Self::Custom(..) => custom_pixels(self.depth(), count, seed),
        }
    }
}

fn layout_float_image(
    width: u32,
    height: u32,
    format: LayoutFloat,
    buffer16: bool,
    wide_extra: bool,
) -> Image {
    let count = width as usize * height as usize;
    let image = Image::new(
        width,
        height,
        PixelLayout::Rgba,
        format.pixels(count * 4, 0),
    )
    .unwrap();
    let mut metadata = image.metadata().clone();
    metadata.bit_depth = format.depth();
    metadata.modular_16_bit_buffer_sufficient = buffer16;
    for alpha in &mut metadata.extra_channels {
        alpha.bit_depth = format.depth();
    }
    let image = image.with_metadata(metadata, vec![]).unwrap();
    let extra = if wide_extra {
        LayoutFloat::Single
    } else {
        format
    };
    with_extras(image, vec![(extra.pixels(count, 7), extra.depth())])
}

fn global_layout_options() -> EncodeOptions {
    // Predictor/entropy search is covered by active_*; these cases isolate
    // full-frame transforms, partitions and source tree/header orchestration.
    let mut options = direct_options(Coding::Source);
    options.threads = 2;
    options.group_size_shift = 0;
    options.responsive = true;
    options.passes = Passes::progressive();
    options.modular.rct = Some(6);
    options.modular.predictor = Predictor::Gradient;
    options.modular.fixed_tree = Some(FixedTree::GradientLeaf);
    options.modular.source_tree_entropy = true;
    options
}

// Check fixture eligibility independently from private orchestration internals.
// Existing internal tests inspect transform scripts; these official tests
// exercise the full public encode path and compare every decoded float word.
fn assert_layout_gate_inputs(image: &Image, options: &EncodeOptions, expected: &[u32]) {
    assert_eq!(options.modular.rct, Some(6));
    let limit = if image.metadata().modular_16_bit_buffer_sufficient {
        16
    } else {
        32
    };
    let mut depth = (0..image.channel_count())
        .map(|c| {
            let depth = image.channel_depth(c);
            assert!(depth.is_float());
            u32::from(depth.bits_per_sample)
        })
        .max()
        .unwrap();
    let mut eligible = Vec::new();
    if depth + 1 < limit {
        eligible.push(0);
        depth += 1;
    }
    if options.responsive && depth + 2 < limit {
        eligible.push(2);
    }
    assert_eq!(eligible, expected);
}

#[test]
#[ignore = "requires official djxl; responsive progressive float group/DC boundaries"]
fn global_layout_finite_float_progressive_word_wire() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for (width, height) in [(257, 17), (17, 257), (1153, 2)] {
        for format in [
            LayoutFloat::Half,
            LayoutFloat::Single,
            LayoutFloat::Custom(24, 8),
            LayoutFloat::Custom(16, 5),
            LayoutFloat::Custom(10, 5),
        ] {
            let image = layout_float_image(width, height, format, false, false);
            let mut options = global_layout_options();
            let expected: &[u32] = if matches!(format, LayoutFloat::Single) {
                &[]
            } else {
                &[0, 2]
            };
            assert_layout_gate_inputs(&image, &options, expected);
            for shared in [false, true] {
                options.shared_tree = shared;
                check_image_with_options(
                    &image,
                    &options,
                    &format!(
                        "global-layout/{format:?}/{width}x{height}/progressive/shared={shared}"
                    ),
                    &decoders,
                    &scratch,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires official djxl; exact float words across RCT/squeeze headroom gates"]
fn global_layout_float_headroom_gates_word_wire() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let cases: &[(LayoutFloat, bool, bool, &[u32])] = &[
        (LayoutFloat::Half, false, false, &[0, 2]),
        (LayoutFloat::Half, true, false, &[]),
        (LayoutFloat::Half, false, true, &[]),
        (LayoutFloat::Single, false, false, &[]),
        (LayoutFloat::Custom(28, 8), false, false, &[0, 2]),
        (LayoutFloat::Custom(29, 8), false, false, &[0]),
        (LayoutFloat::Custom(30, 8), false, false, &[0]),
        (LayoutFloat::Custom(31, 8), false, false, &[]),
        (LayoutFloat::Custom(12, 5), true, false, &[]),
        (LayoutFloat::Custom(13, 5), true, false, &[]),
    ];
    for &(format, buffer16, wide_extra, expected) in cases {
        let image = layout_float_image(257, 9, format, buffer16, wide_extra);
        let options = global_layout_options();
        if buffer16 {
            // The public API permits this promise only for integer channels
            // of at most 12 bits, including when float words are narrower.
            let error = encode(&image, &options).expect_err("float buffer16 promise must fail");
            assert!(
                matches!(&error, rs_jxl::Error::InvalidInput(message)
                    if message == "16-bit Modular buffer promise requires at most 12-bit integer input channels"),
                "{format:?}: unexpected validation error: {error}"
            );
            continue;
        }
        assert_layout_gate_inputs(&image, &options, expected);
        check_image_with_options(
            &image,
            &options,
            &format!("global-layout/gates/{format:?}/buffer16={buffer16}/wide-extra={wide_extra}"),
            &decoders,
            &scratch,
        );
    }
}

#[test]
#[ignore = "requires official djxl; float frame-area streaming across ten AC groups"]
fn global_layout_frame_area_streaming_float_word_wire() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for format in [
        LayoutFloat::Half,
        LayoutFloat::Single,
        LayoutFloat::Custom(24, 8),
        LayoutFloat::Custom(16, 5),
    ] {
        let image = layout_float_image(1153, 2, format, false, false);
        let mut options = global_layout_options();
        options.responsive = false;
        options.passes = Passes::default();
        options.shared_tree = true;
        options.modular.palette_policy = Some(PaletteOptions {
            groupwise: true,
            palette_colors: 0,
            channel_colors_pre_transform_percent: 0.0,
            channel_colors_percent: 0.0,
            ..Default::default()
        });
        assert_eq!(
            groupwise_decision(
                &image,
                &options.modular,
                options.group_size_shift,
                &options.passes,
                options.responsive,
                false,
            )
            .unwrap(),
            GroupwiseDecision::FrameAreas
        );
        let expected: &[u32] = if matches!(format, LayoutFloat::Single) {
            &[]
        } else {
            &[0]
        };
        assert_layout_gate_inputs(&image, &options, expected);
        check_image_with_options(
            &image,
            &options,
            &format!("global-layout/frame-areas/{format:?}/1153x2"),
            &decoders,
            &scratch,
        );
    }
}

fn active_f32_image(width: u32, height: u32) -> Image {
    let count = width as usize * height as usize;
    let base = Image::new(
        width,
        height,
        PixelLayout::Rgba,
        finite_pixels(false, count * 4, 0),
    )
    .unwrap();
    let depth = BitDepth::floating(32, 8).unwrap();
    // Two trailing constant planes exercise LearnTree's exact offset leaves,
    // including offset i32::MIN from -0.0. The variable planes repeatedly
    // cross signed-word boundaries and traverse every weighted row-ring slot.
    with_extras(
        base,
        vec![
            (finite_pixels(false, count, 9), depth),
            (PixelData::F32(vec![-3.25; count]), depth),
            (PixelData::F32(vec![-0.0; count]), depth),
        ],
    )
}

fn active_options(
    image: &Image,
    source_sampling: bool,
    predictor: Predictor,
    rct: Option<u32>,
) -> EncodeOptions {
    let mut options = EncodeOptions::default();
    options.modular.predictor = predictor;
    options.modular.rct = rct;
    assert!(options.modular.fixed_tree.is_none());
    if source_sampling {
        // Source sampling requests complete beneficial-split search. The
        // default encoder leaf budget is a speed policy, not sufficient for
        // this deliberately adversarial full-word interoperability corpus.
        options.modular.max_tree_leaves = rs_jxl::modular::ma_train::MAX_LEAVES;
        options.modular.sample_limit = (image.width() as usize)
            .checked_mul(image.height() as usize)
            .and_then(|n| n.checked_mul(image.channel_count()))
            .expect("bounded active fixture population");
        let group = 128usize << options.group_size_shift;
        let width = image.width() as usize;
        let height = image.height() as usize;
        let groups = width.div_ceil(group) * height.div_ceil(group);
        let dc_groups = width.div_ceil(group * 8) * height.div_ceil(group * 8);
        options.modular.sampling = Some(
            SamplingPolicy::from_source(&SourcePolicy {
                predictor,
                num_streams: if groups == 1 {
                    1
                } else {
                    18 + 3 * dc_groups + groups
                },
                num_extra_channels: image.metadata().extra_channels.len(),
                is_gray: image.color_channels() == 1,
                max_previous_channels: options.modular.previous_channels,
                ..SourcePolicy::default()
            })
            .unwrap(),
        );
    }
    options
}

#[test]
#[ignore = "requires official djxl; adaptive wide-residual F32 integration"]
fn active_f32_default_adaptive_legacy_and_source_sampling() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for (width, height) in [(37, 19), (259, 19)] {
        let image = active_f32_image(width, height);
        for source in [false, true] {
            for rct in [None, Some(6)] {
                let options = active_options(&image, source, Predictor::Best, rct);
                check_image_with_options(
                    &image,
                    &options,
                    &format!(
                        "active/default-adaptive/source={source}/rct={rct:?}/{width}x{height}"
                    ),
                    &decoders,
                    &scratch,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires official djxl; all 14 predictors with wrapped F32 residuals"]
fn active_f32_all_predictors_no_rct_and_rct6() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let image = active_f32_image(37, 19);
    for id in 0..14 {
        let predictor = Predictor::try_from(id).unwrap();
        for source in [false, true] {
            for rct in [None, Some(6)] {
                // A one-element predictor candidate list forces this predictor
                // in all nonconstant leaves without disabling MA learning.
                let options = active_options(&image, source, predictor, rct);
                check_image_with_options(
                    &image,
                    &options,
                    &format!("active/{predictor:?}/source={source}/rct={rct:?}"),
                    &decoders,
                    &scratch,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires official djxl; weighted full-row-ring state and all presets"]
fn active_f32_weighted_full_ring_all_presets() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let image = active_f32_image(37, 19);
    for preset in 0..=4 {
        for source in [false, true] {
            for rct in [None, Some(6)] {
                let mut options = active_options(&image, source, Predictor::Weighted, rct);
                options.modular.weighted_preset = preset;
                check_image_with_options(
                    &image,
                    &options,
                    &format!("active/weighted-ring/preset={preset}/source={source}/rct={rct:?}"),
                    &decoders,
                    &scratch,
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LeafSpec {
    predictor: Predictor,
    offset: i32,
    multiplier: u32,
}

const CUSTOM_LEAVES: [LeafSpec; 6] = [
    LeafSpec {
        predictor: Predictor::Zero,
        offset: i32::MIN,
        multiplier: 1,
    },
    LeafSpec {
        predictor: Predictor::Zero,
        offset: i32::MIN,
        multiplier: 3,
    },
    LeafSpec {
        predictor: Predictor::Left,
        offset: 0,
        multiplier: 3,
    },
    LeafSpec {
        predictor: Predictor::Left,
        offset: i32::MAX,
        multiplier: 3,
    },
    LeafSpec {
        predictor: Predictor::Left,
        offset: i32::MIN,
        multiplier: 2,
    },
    LeafSpec {
        predictor: Predictor::Top,
        offset: 3,
        multiplier: 6,
    },
];

fn custom_multiplier_tree() -> Tree {
    let mut nodes = Vec::new();
    for (channel, spec) in CUSTOM_LEAVES.iter().enumerate() {
        if channel + 1 < CUSTOM_LEAVES.len() {
            let index = nodes.len();
            nodes.push(Node::Split {
                property: 0,
                threshold: channel as i32,
                greater: index + 2,
                less_equal: index + 1,
            });
        }
        nodes.push(Node::Leaf {
            predictor: spec.predictor,
            offset: spec.offset,
            multiplier: spec.multiplier,
        });
    }
    Tree::new(nodes).unwrap()
}

// Construct finite F32 WORDS in a chosen residue class. This adjusts fixture
// words, never encoded image data, and does not perform float arithmetic.
fn congruent_finite_word(target: u32, origin: i64, multiplier: u32) -> i32 {
    let target = i64::from(target as i32);
    let multiplier = i64::from(multiplier);
    let lower = target - (target - origin).rem_euclid(multiplier);
    for candidate in [lower, lower + multiplier] {
        if let Ok(word) = i32::try_from(candidate)
            && f32::from_bits(word as u32).is_finite()
        {
            return word;
        }
    }
    panic!("fixture cannot align word {target} to {origin} modulo {multiplier}");
}

fn custom_multiplier_image(width: u32, height: u32) -> Image {
    let w = width as usize;
    let h = height as usize;
    let mut planes = vec![vec![0i32; w * h]; CUSTOM_LEAVES.len()];
    let targets = [
        0x8000_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x0000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x0080_0000,
        0x8080_0000,
        0x8000_0001,
        0x0000_0001,
    ];
    // The oracle below uses default 256-pixel groups, with predictor state
    // reset per group even when the MA tree is shared by the whole frame.
    for (channel, spec) in CUSTOM_LEAVES.iter().enumerate() {
        let plane = &mut planes[channel];
        for gy in (0..h).step_by(256) {
            for gx in (0..w).step_by(256) {
                for y in gy..(gy + 256).min(h) {
                    for x in gx..(gx + 256).min(w) {
                        let index = y * w + x;
                        let left = if x > gx {
                            plane[index - 1]
                        } else if y > gy {
                            plane[index - w]
                        } else {
                            0
                        };
                        let top = if y > gy { plane[index - w] } else { left };
                        let prediction = match spec.predictor {
                            Predictor::Zero => 0,
                            Predictor::Left => i64::from(left),
                            Predictor::Top => i64::from(top),
                            _ => unreachable!(),
                        };
                        let target = targets[(x + y * 3 + channel) % targets.len()];
                        plane[index] = congruent_finite_word(
                            target,
                            prediction + i64::from(spec.offset),
                            spec.multiplier,
                        );
                    }
                }
            }
        }
    }
    let pixels = PixelData::F32(
        (0..w * h)
            .flat_map(|pixel| (0..4).map(move |channel| (pixel, channel)))
            .map(|(pixel, channel)| f32::from_bits(planes[channel][pixel] as u32))
            .collect(),
    );
    let base = Image::new(width, height, PixelLayout::Rgba, pixels).unwrap();
    with_extras(
        base,
        planes[4..]
            .iter()
            .map(|plane| {
                (
                    PixelData::F32(
                        plane
                            .iter()
                            .map(|&word| f32::from_bits(word as u32))
                            .collect(),
                    ),
                    BitDepth::floating(32, 8).unwrap(),
                )
            })
            .collect(),
    )
}

#[derive(Debug, Default)]
struct DivisionProof {
    samples: usize,
    wide_prediction_residuals: usize,
    wide_offset_residuals: usize,
    wide_quotients: usize,
    early_narrowing_breaks_divisibility: usize,
}

fn prove_custom_divisibility(image: &Image, tree: &Tree) -> Vec<DivisionProof> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    assert_eq!(image.layout(), PixelLayout::Rgba);
    assert_eq!(image.channel_count(), CUSTOM_LEAVES.len());
    let PixelData::F32(interleaved) = image.pixels() else {
        unreachable!()
    };
    let mut proofs = Vec::new();
    for (channel, spec) in CUSTOM_LEAVES.iter().enumerate() {
        let (_, predictor, offset, multiplier) = tree.lookup(&[channel as i32]).unwrap();
        assert_eq!(
            (predictor, offset, multiplier),
            (spec.predictor, spec.offset, spec.multiplier)
        );
        let words: Vec<i32> = if channel < 4 {
            interleaved
                .iter()
                .skip(channel)
                .step_by(4)
                .map(|v| v.to_bits() as i32)
                .collect()
        } else {
            let PixelData::F32(values) = &image.extra_planes()[channel - 4] else {
                unreachable!()
            };
            values.iter().map(|v| v.to_bits() as i32).collect()
        };
        let mut proof = DivisionProof::default();
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                let left_index = if x % 256 != 0 {
                    Some(index - 1)
                } else if y % 256 != 0 {
                    Some(index - width)
                } else {
                    None
                };
                let prediction_index = match predictor {
                    Predictor::Zero => None,
                    Predictor::Left => left_index,
                    Predictor::Top if y % 256 != 0 => Some(index - width),
                    Predictor::Top => left_index,
                    _ => unreachable!(),
                };
                let prediction = prediction_index.map_or(0, |i| i64::from(words[i]));
                let residual = i64::from(words[index]) - prediction;
                let adjusted = residual - i64::from(offset);
                let multiplier = i64::from(multiplier);
                assert_eq!(
                    adjusted % multiplier,
                    0,
                    "channel={channel}, x={x}, y={y}: wide {residual} - offset {offset} must be divisible by {multiplier}"
                );
                let quotient = adjusted / multiplier;
                // Source decoder reconstruction is modulo 2^32, including a
                // quotient that only becomes signed i32 at PackSigned.
                let reconstructed = (i64::from(quotient as i32) * multiplier
                    + i64::from(offset)
                    + prediction) as i32;
                assert_eq!(reconstructed, words[index]);
                assert!(f32::from_bits(words[index] as u32).is_finite());
                proof.samples += 1;
                proof.wide_prediction_residuals += usize::from(i32::try_from(residual).is_err());
                proof.wide_offset_residuals += usize::from(i32::try_from(adjusted).is_err());
                proof.wide_quotients += usize::from(i32::try_from(quotient).is_err());
                proof.early_narrowing_breaks_divisibility +=
                    usize::from(i64::from(adjusted as i32) % multiplier != 0);
            }
        }
        assert_eq!(proof.samples, width * height);
        assert!(
            proof.wide_offset_residuals > 0,
            "channel {channel}: {proof:?}"
        );
        if spec.predictor != Predictor::Zero {
            assert!(
                proof.wide_prediction_residuals > 0,
                "channel {channel}: {proof:?}"
            );
        }
        if spec.multiplier % 3 == 0 {
            assert!(
                proof.early_narrowing_breaks_divisibility > 0,
                "channel {channel}: {proof:?}"
            );
        }
        if channel == 0 || channel == 4 {
            assert!(proof.wide_quotients > 0, "channel {channel}: {proof:?}");
        }
        proofs.push(proof);
    }
    proofs
}

fn custom_tree_options(coding: Coding, tree: &Tree) -> EncodeOptions {
    let mut options = direct_options(coding);
    options.modular.fixed_tree = None;
    options.modular.custom_tree = Some(tree.clone());
    options.modular.max_tree_leaves = tree.leaves();
    options
}

#[test]
#[ignore = "requires official djxl; explicit wide offset/division before PackSigned"]
fn active_f32_custom_tree_wide_offset_and_odd_multipliers() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let tree = custom_multiplier_tree();
    for (width, height) in [(37, 11), (259, 5)] {
        let image = custom_multiplier_image(width, height);
        let proofs = prove_custom_divisibility(&image, &tree);
        eprintln!("custom-tree {width}x{height} per-channel division proof: {proofs:?}");
        for coding in [Coding::Prefix, Coding::Ans] {
            for shared_tree in [false, true] {
                for source_tree_entropy in [false, true] {
                    let mut options = custom_tree_options(coding, &tree);
                    options.shared_tree = shared_tree;
                    options.modular.source_tree_entropy = source_tree_entropy;
                    check_image_with_options(
                        &image,
                        &options,
                        &format!(
                            "active/custom-tree/{width}x{height}/{coding:?}/shared={shared_tree}/source-tree={source_tree_entropy}"
                        ),
                        &decoders,
                        &scratch,
                    );
                }
            }
        }
    }
}

fn check_nonfinite(half: bool, nan: bool) {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let words: &[u32] = match (half, nan) {
        (true, false) => &[0x7c00, 0xfc00, 0, 0x8000],
        (true, true) => &[0x7e01, 0xfe55, 0x7c01, 0xfc01, 0x7fff, 0xffff],
        (false, false) => &[0x7f80_0000, 0xff80_0000, 0, 0x8000_0000],
        (false, true) => &[
            0x7fc1_2345,
            0xffc5_4321,
            0x7f80_0001,
            0xff80_0001,
            0x7fff_ffff,
            0xffff_ffff,
        ],
    };
    let pixels = |count: usize, seed: usize| {
        if half {
            PixelData::F16(
                (0..count)
                    .map(|i| words[(i + seed) % words.len()] as u16)
                    .collect(),
            )
        } else {
            PixelData::F32(
                (0..count)
                    .map(|i| f32::from_bits(words[(i + seed) % words.len()]))
                    .collect(),
            )
        }
    };
    for layout in [PixelLayout::Gray, PixelLayout::Rgba] {
        let base = Image::new(17, 1, layout, pixels(17 * layout.channels(), 0)).unwrap();
        let extra = pixels(17, 1);
        let depth = extra.bit_depth();
        let image = with_extras(base, vec![(extra, depth)]);
        check_image(
            &image,
            &format!("EXPERIMENTAL/half={half}/nan={nan}/{layout:?}"),
            &decoders,
            &scratch,
        );
    }
}

#[test]
#[ignore = "experimental F16 infinity/signed-zero words; explicit run must pass official djxl"]
fn experimental_f16_infinity_word_wire() {
    check_nonfinite(true, false);
}

#[test]
#[ignore = "experimental F16 quiet/signaling NaN payloads; canonicalization is a failure"]
fn experimental_f16_nan_payload_word_wire() {
    check_nonfinite(true, true);
}

#[test]
#[ignore = "experimental F32 infinity/signed-zero words; explicit run must pass official djxl"]
fn experimental_f32_infinity_word_wire() {
    check_nonfinite(false, false);
}

#[test]
#[ignore = "experimental F32 quiet/signaling NaN payloads; canonicalization is a failure"]
fn experimental_f32_nan_payload_word_wire() {
    check_nonfinite(false, true);
}

fn npy_fixture(shape: &str, words: &[u32]) -> Vec<u8> {
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape}), }}\n");
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for &word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[test]
fn npy_parser_checks_header_shape_and_exact_payload() {
    let words = [0, 0x8000_0000, 0x7fc1_2345, 0xff80_0000];
    let bytes = npy_fixture("1, 1, 2, 2", &words);
    let parsed = parse_npy(&bytes).unwrap();
    assert_eq!(parsed.shape, [1, 1, 2, 2]);
    assert_eq!(parsed.words, words);
    for end in [0, 7, 9, bytes.len() - 1] {
        assert!(parse_npy(&bytes[..end]).is_err());
    }
    let mut extra_byte = bytes.clone();
    extra_byte.push(0);
    assert!(parse_npy(&extra_byte).is_err());
    let mut version_two = bytes.clone();
    version_two[6] = 2;
    assert!(parse_npy(&version_two).is_err());
    for shape in [
        "1, 2, 2",
        "1, 0, 2, 2",
        "1, 1, 3, 2",
        "1, -1, 2, 2",
        "1, 1, 2, 2, 1",
    ] {
        assert!(parse_npy(&npy_fixture(shape, &words)).is_err());
    }
    assert!(parse_npy(&npy_fixture(&format!("1, {}, 2, 2", usize::MAX), &[])).is_err());
    for (needle, replacement) in [(b"<f4".as_slice(), b"<f8".as_slice()), (b"False", b"True ")] {
        let mut malformed = bytes.clone();
        let offset = malformed
            .windows(needle.len())
            .position(|w| w == needle)
            .unwrap();
        malformed[offset..offset + needle.len()].copy_from_slice(replacement);
        assert!(parse_npy(&malformed).is_err());
    }
}

#[test]
fn half_widening_known_words_and_finite_fixture_domains() {
    for (half, single) in [
        (0x0000, 0x0000_0000),
        (0x8000, 0x8000_0000),
        (0x0001, 0x3380_0000),
        (0x03ff, 0x387f_c000),
        (0x0400, 0x3880_0000),
        (0x3c00, 0x3f80_0000),
        (0xbc00, 0xbf80_0000),
        (0x7bff, 0x477f_e000),
        (0x7c00, 0x7f80_0000),
        (0xfc00, 0xff80_0000),
        (0x7e01, 0x7fc0_2000),
        (0xfc01, 0xff80_2000),
    ] {
        assert_eq!(half_to_f32_bits(half), single);
    }
    assert!(HALF_FINITE.iter().all(|&v| (v >> 10) & 31 != 31));
    assert!(FLOAT_FINITE.iter().all(|&v| f32::from_bits(v).is_finite()));
    assert!(HALF_FINITE.contains(&0x8000));
    assert!(FLOAT_FINITE.contains(&0x8000_0000));
}

#[test]
fn integer_projection_precision_limits_are_not_integer_recovery() {
    let value = 8_388_607;
    let projected = integer_projection_bits(value, 24, true);
    assert_eq!(projected, 0x3f00_0000); // exact failure reported by official djxl
    assert_eq!(
        (f64::from(f32::from_bits(projected)) * 16_777_215.0).round(),
        8_388_608.0
    );
    assert_eq!(integer_projection_bits(value, 24, false), 0x3eff_ffff);
    let maximum = (1u32 << 31) - 1;
    assert_eq!(
        integer_projection_bits(maximum, 31, false),
        integer_projection_bits(maximum - 1, 31, false)
    );
}

#[test]
fn custom_float_metadata_rejects_nonrepresentable_values_in_each_plane_kind() {
    for (bits, exponent) in [(24, 8), (16, 5), (10, 5), (8, 3)] {
        let depth = BitDepth::floating(bits, exponent).unwrap();
        let values = custom_float_values(depth);
        let base = Image::new(
            values.len() as u32,
            1,
            PixelLayout::Gray,
            PixelData::F32(values),
        )
        .unwrap();
        let mut metadata = base.metadata().clone();
        metadata.bit_depth = depth;
        assert!(
            base.with_metadata(metadata, vec![]).is_ok(),
            "representable {bits}/{exponent}"
        );
        let bad = f32::from_bits(0x3f80_0001); // 1 + 2^-23 cannot fit any of these
        let base = Image::new(1, 1, PixelLayout::Gray, PixelData::F32(vec![bad])).unwrap();
        let mut metadata = base.metadata().clone();
        metadata.bit_depth = depth;
        assert!(base.with_metadata(metadata, vec![]).is_err());

        let base =
            Image::new(1, 1, PixelLayout::GrayAlpha, PixelData::F32(vec![1.0, bad])).unwrap();
        let mut metadata = base.metadata().clone();
        metadata.extra_channels[0].bit_depth = depth;
        assert!(base.with_metadata(metadata, vec![]).is_err());

        let base = Image::new(1, 1, PixelLayout::Gray, PixelData::F32(vec![1.0])).unwrap();
        let mut metadata = base.metadata().clone();
        let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
        extra.bit_depth = depth;
        metadata.extra_channels.push(extra);
        assert!(
            base.with_metadata(metadata, vec![PixelData::F32(vec![bad])])
                .is_err()
        );
    }
    // This value is numerically representable as a 24/8 subnormal, but the
    // source encoder's implicit-one step does not preserve its source inverse.
    let base = Image::new(
        1,
        1,
        PixelLayout::Gray,
        PixelData::F32(vec![f32::from_bits(0x0000_0200)]),
    )
    .unwrap();
    let mut metadata = base.metadata().clone();
    metadata.bit_depth = BitDepth::floating(24, 8).unwrap();
    assert!(base.with_metadata(metadata, vec![]).is_err());
}

#[test]
fn active_fixture_crosses_signed_word_boundaries_and_keeps_learning_enabled() {
    let image = active_f32_image(37, 19);
    let PixelData::F32(values) = image.pixels() else {
        unreachable!()
    };
    let mut wide = false;
    for channel in 0..image.layout().channels() {
        let words: Vec<_> = values
            .iter()
            .skip(channel)
            .step_by(image.layout().channels())
            .map(|v| v.to_bits() as i32)
            .collect();
        wide |= words
            .windows(2)
            .any(|pair| i32::try_from(i64::from(pair[1]) - i64::from(pair[0])).is_err());
    }
    assert!(wide, "fixture must exercise a residual outside signed i32");
    for source in [false, true] {
        let options = active_options(&image, source, Predictor::Best, Some(6));
        assert!(options.modular.fixed_tree.is_none());
        assert_eq!(options.modular.sampling.is_some(), source);
        assert!(options.modular.max_tree_leaves > 3);
        if source {
            assert_eq!(
                options.modular.max_tree_leaves,
                rs_jxl::modular::ma_train::MAX_LEAVES
            );
            assert_eq!(
                options.modular.sample_limit,
                image.width() as usize * image.height() as usize * image.channel_count()
            );
        }
        options.modular.validate().unwrap();
    }
}

#[test]
fn custom_tree_fixture_proves_wide_divisibility_and_constant_residue_classes() {
    let tree = custom_multiplier_tree();
    assert_eq!(tree.leaves(), CUSTOM_LEAVES.len());
    for (width, height) in [(37, 11), (259, 5)] {
        let image = custom_multiplier_image(width, height);
        prove_custom_divisibility(&image, &tree);
        let PixelData::F32(values) = image.pixels() else {
            unreachable!()
        };
        for pixel in values.chunks_exact(4) {
            // Zero/offset=MIN/m=3 and Left/offset=0/m=3 have constant
            // signed-word residue classes, including the first group pixel.
            assert_eq!(
                (pixel[1].to_bits() as i32).rem_euclid(3),
                i32::MIN.rem_euclid(3)
            );
            assert_eq!((pixel[2].to_bits() as i32).rem_euclid(3), 0);
        }
    }
}

#[test]
fn custom_tree_rejects_conflicting_modes_bad_multipliers_and_indivisible_data() {
    let tree = custom_multiplier_tree();
    let mut options = custom_tree_options(Coding::Ans, &tree);
    options.modular.validate().unwrap();
    options.modular.fixed_tree = Some(FixedTree::Zero);
    assert!(options.modular.validate().is_err());
    for multiplier in [0, 1u32 << 31, u32::MAX] {
        assert!(
            Tree::new(vec![Node::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier
            }])
            .is_err()
        );
    }

    let image = custom_multiplier_image(9, 3);
    let PixelData::F32(mut values) = image.pixels().clone() else {
        unreachable!()
    };
    // Corrupt the m=3 channel by exactly one signed-word unit. The F32 value
    // remains finite, but the encoder must reject rather than truncate /3.
    values[1] = f32::from_bits(values[1].to_bits().wrapping_add(1));
    assert!(values[1].is_finite());
    let bad = Image::new(9, 3, PixelLayout::Rgba, PixelData::F32(values))
        .unwrap()
        .with_metadata(image.metadata().clone(), image.extra_planes().to_vec())
        .unwrap();
    let error = encode(&bad, &custom_tree_options(Coding::Ans, &tree)).unwrap_err();
    assert!(
        matches!(error, rs_jxl::Error::Encoding(message) if message.contains("does not divide"))
    );
}
