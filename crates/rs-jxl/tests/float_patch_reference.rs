//! Independent official-djxl oracle for actual floating-color SourceAdd frames.
//!
//! Set RS_JXL_DJXL to an OS path list of official decoder executables. The
//! ignored test requires every decoder to succeed; it never skips failures or
//! accepts NaN canonicalization, signed-zero changes, or approximate samples.
//! No other test file or native decoder is included. Harness conventions match
//! float_reference.rs and metadata_frame_reference.rs; the fixed NPY schema is
//! the one emitted by frozen b866b3d lib/extras/enc/npy.cc.
//!
//! Six bounded images cover F16 G/GA/RGB/RGBA and custom 16/6 and 10/5 formats.
//! Each checks patches-off versus active SourceAdd, AND original float words
//! using independent, injective IEEE widening. Subsampled extras necessarily
//! pass through the official upsampler: compare their complete output words
//! against the patches-off decode, not an invented interpolation or a claim
//! that displayed full-resolution samples equal native subsampled words.

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
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::modular::sample_convert::SampleFormat;
use rs_jxl::tools::patches::{self, FloatImageView, PatchBlendMode, SourcePatchConfig};
use rs_jxl::{
    EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout, encode, encode_to,
};

const WIDTH: usize = 129;
const HEIGHT: usize = 83;
const ORIGINS: [(usize, usize); 4] = [(17, 17), (81, 17), (33, 53), (97, 53)];
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug)]
struct Case {
    bits: u8,
    exponent: u8,
    layout: PixelLayout,
    mixed_extras: bool,
}

fn cases() -> [Case; 6] {
    [
        Case {
            bits: 16,
            exponent: 5,
            layout: PixelLayout::Gray,
            mixed_extras: false,
        },
        Case {
            bits: 16,
            exponent: 5,
            layout: PixelLayout::GrayAlpha,
            mixed_extras: false,
        },
        Case {
            bits: 16,
            exponent: 5,
            layout: PixelLayout::Rgb,
            mixed_extras: false,
        },
        Case {
            bits: 16,
            exponent: 5,
            layout: PixelLayout::Rgba,
            mixed_extras: true,
        },
        Case {
            bits: 16,
            exponent: 6,
            layout: PixelLayout::Gray,
            mixed_extras: false,
        },
        Case {
            bits: 10,
            exponent: 5,
            layout: PixelLayout::Rgba,
            mixed_extras: true,
        },
    ]
}

// Independent widening, not the encoder's float_to_int/int_to_float helpers.
// These cases have exponent_bits < 8, so every small-format subnormal becomes
// a normal f32. Integer construction preserves signaling-NaN payloads too.
fn widened_word(word: u32, bits: u8, exponent_bits: u8) -> u32 {
    assert!((5..=16).contains(&bits) && (2..8).contains(&exponent_bits));
    let mantissa_bits = u32::from(bits - exponent_bits - 1);
    assert!(mantissa_bits >= 2 && word < (1u32 << bits));
    let sign = (word >> (bits - 1)) << 31;
    let exponent_max = (1u32 << exponent_bits) - 1;
    let exponent = (word >> mantissa_bits) & exponent_max;
    let fraction = word & ((1 << mantissa_bits) - 1);
    let bias = (1i32 << (exponent_bits - 1)) - 1;
    match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = 31 - fraction.leading_zeros();
            let e32 = 127 + 1 - bias - mantissa_bits as i32 + leading as i32;
            sign | ((e32 as u32) << 23) | ((fraction - (1 << leading)) << (23 - leading))
        }
        e if e == exponent_max => sign | 0x7f80_0000 | (fraction << (23 - mantissa_bits)),
        _ => {
            sign | (((exponent as i32 - bias + 127) as u32) << 23)
                | (fraction << (23 - mantissa_bits))
        }
    }
}

fn glyph(x: usize, y: usize) -> bool {
    ORIGINS
        .iter()
        .any(|&(x0, y0)| (x0..x0 + 6).contains(&x) && (y0..y0 + 5).contains(&y))
}

struct Fixture {
    image: Image,
    colors: Vec<Vec<i32>>,
    expected_main: Vec<u32>,
    config: SourcePatchConfig,
    case: Case,
}

impl Fixture {
    fn view(&self) -> FloatImageView<'_> {
        FloatImageView {
            width: WIDTH,
            height: HEIGHT,
            planes: &self.colors,
            format: SampleFormat::Float {
                bits: self.case.bits,
                exponent_bits: self.case.exponent,
            },
        }
    }

    fn assert_active(&self) {
        let plan = patches::discover_source_float(self.view(), &self.config)
            .unwrap_or_else(|e| panic!("{:?}: discovery: {e}", self.case))
            .expect("fixture must activate SourceAdd, not return None");
        assert_eq!(plan.dictionary.positions.len(), 4, "{:?}", self.case);
        assert_eq!(plan.dictionary.references.len(), 1, "{:?}", self.case);
        assert!(
            plan.dictionary
                .blendings
                .iter()
                .all(|b| b.mode == PatchBlendMode::Add)
        );
        assert_eq!(
            plan.dictionary.num_extra_channels, 0,
            "detector is color-only"
        );
        assert_ne!(
            (plan.reference.width, plan.reference.height),
            (WIDTH, HEIGHT)
        );
        assert_eq!(
            patches::compose_float_add(&plan.residual, &plan.reference, &plan.dictionary).unwrap(),
            self.colors
        );
        for position in &plan.dictionary.positions {
            assert!(
                position.y > 0,
                "raw special words must remain outside every patch"
            );
            if self.image.layout().has_alpha() {
                let pixel = position.y as usize * WIDTH + position.x as usize;
                assert_eq!(
                    self.expected_main
                        [pixel * self.image.layout().channels() + self.image.color_channels()],
                    0x8000_0000
                );
                assert!(
                    self.expected_main[pixel * self.image.layout().channels()] & 0x7fff_ffff != 0,
                    "hidden color must be nonzero"
                );
            }
        }
        patches::tokenize(&plan.dictionary).unwrap();
    }
}

fn fixture(case: Case) -> Fixture {
    let color_count = if case.layout.channels() < 3 { 1 } else { 3 };
    let mantissa = u32::from(case.bits - case.exponent - 1);
    let bias = (1u32 << (case.exponent - 1)) - 1;
    let sign = 1u32 << (case.bits - 1);
    let half = (bias - 1) << mantissa;
    let one = bias << mantissa;
    // 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, all exact in every fixture format.
    let levels = [
        half,
        half | (1 << (mantissa - 1)),
        one,
        one | (1 << (mantissa - 2)),
        one | (1 << (mantissa - 1)),
        one | (3 << (mantissa - 2)),
        (bias + 1) << mantissa,
    ];
    let nan = ((1u32 << case.exponent) - 1) << mantissa;
    let quiet = 1 << (mantissa - 1);
    let specials = [sign, nan | 1, nan | quiet | 3, sign | nan | quiet | 1, 1];
    let mut colors: Vec<_> = (0..color_count)
        .map(|_| Vec::with_capacity(WIDTH * HEIGHT))
        .collect();
    let mut words = Vec::with_capacity(WIDTH * HEIGHT * case.layout.channels());
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let level = 2 * usize::from(x >= WIDTH / 2) + if glyph(x, y) { 0 } else { 2 };
            for (channel, plane) in colors.iter_mut().enumerate() {
                let word = if y == 0 && x < specials.len() {
                    specials[x]
                } else {
                    levels[level + channel]
                };
                plane.push(word as i32);
                words.push(word);
            }
            if case.layout.has_alpha() {
                words.push(if glyph(x, y) {
                    sign
                } else {
                    [0, sign, one, half][(x + y) % 4]
                });
            }
        }
    }
    let expected_main: Vec<_> = words
        .iter()
        .map(|&w| widened_word(w, case.bits, case.exponent))
        .collect();
    let pixels = if (case.bits, case.exponent) == (16, 5) {
        PixelData::F16(words.into_iter().map(|w| w as u16).collect())
    } else {
        PixelData::F32(expected_main.iter().map(|&w| f32::from_bits(w)).collect())
    };
    let image = Image::new(WIDTH as u32, HEIGHT as u32, case.layout, pixels).unwrap();
    let mut metadata = image.metadata().clone();
    metadata.bit_depth = BitDepth::floating(case.bits, case.exponent).unwrap();
    for extra in &mut metadata.extra_channels {
        extra.bit_depth = metadata.bit_depth;
    }
    let mut extras = Vec::new();
    if case.mixed_extras {
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
            info.name = format!("float-patch-extra-{}", extras.len());
            metadata.extra_channels.push(info);
            let count = WIDTH.div_ceil(1 << shift) * HEIGHT.div_ceil(1 << shift);
            extras.push(match (depth.bits_per_sample, depth.is_float()) {
                (16, true) => PixelData::F16(
                    (0..count)
                        .map(|i| [0, 0x3400, 0x3800, 0x3c00][i % 4])
                        .collect(),
                ),
                (32, true) => PixelData::F32(
                    (0..count)
                        .map(|i| {
                            f32::from_bits(
                                [
                                    0x8000_0000,
                                    0x7f80_0123,
                                    0x7fc1_2345,
                                    0xffc0_0023,
                                    1,
                                    0x3f00_0000,
                                ][i % 6],
                            )
                        })
                        .collect(),
                ),
                (8, false) => PixelData::U8((0..count).map(|i| (i % 253) as u8).collect()),
                (_, false) => {
                    let mask = (1u32 << depth.bits_per_sample) - 1;
                    PixelData::U16(
                        (0..count)
                            .map(|i| ((i as u32 * 7919) & mask) as u16)
                            .collect(),
                    )
                }
                _ => unreachable!(),
            });
        }
    }
    Fixture {
        image: image.with_metadata(metadata, extras).unwrap(),
        colors,
        expected_main,
        case,
        config: SourcePatchConfig {
            speed_tier: -1,
            floating_point: true,
            ..Default::default()
        },
    }
}

fn options(shared: bool) -> EncodeOptions {
    EncodeOptions {
        threads: 1,
        max_output_bytes: 8 << 20,
        group_size_shift: if shared { 0 } else { 1 },
        shared_tree: shared,
        patches: None,
        source_patches: None,
        modular: ModularOptions {
            predictor: Predictor::Gradient,
            fixed_tree: Some(FixedTree::GradientLeaf),
            rct: None,
            lz77: false,
            entropy: if shared { Coding::Ans } else { Coding::Prefix },
            tree_samples: 1,
            max_tree_leaves: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Npy {
    shape: [usize; 4],
    words: Vec<u32>,
}

// Strict parser for the fixed official NPY dictionary, not arbitrary Python.
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
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect(),
    })
}

fn assert_original(actual: &Npy, fixture: &Fixture, context: &str) {
    let image = &fixture.image;
    let channels = image.channel_count();
    assert_eq!(
        actual.shape,
        [1, HEIGHT, WIDTH, channels],
        "{context}: atlas must not become a visible frame"
    );
    let interleaved = image.layout().channels();
    for (pixel, expected) in fixture.expected_main.chunks_exact(interleaved).enumerate() {
        for (channel, &word) in expected.iter().enumerate() {
            let decoded = actual.words[pixel * channels + channel];
            assert_eq!(
                decoded,
                word,
                "{context}: original main ({},{}) channel {channel}: {decoded:08x} != {word:08x}",
                pixel % WIDTH,
                pixel / WIDTH
            );
        }
    }
    for (extra, data) in image.extra_planes().iter().enumerate() {
        let channel = interleaved + extra;
        if image.channel_shift(channel) != 0 {
            continue;
        }
        for (pixel, decoded) in actual.words.chunks_exact(channels).enumerate() {
            let word = decoded[channel];
            let expected = match data {
                PixelData::F16(values) => widened_word(u32::from(values[pixel]), 16, 5),
                PixelData::F32(values) => values[pixel].to_bits(),
                PixelData::U16(values) => {
                    let value = f32::from_bits(word);
                    assert!(
                        value.is_finite() && (0.0..=1.0).contains(&value),
                        "{context}: extra {extra}"
                    );
                    let maximum = (1u32 << image.channel_depth(channel).bits_per_sample) - 1;
                    assert_eq!(
                        (f64::from(value) * f64::from(maximum)).round(),
                        f64::from(values[pixel]),
                        "{context}: original extra {extra} pixel {pixel}"
                    );
                    continue;
                }
                _ => panic!("unexpected full-resolution extra fixture"),
            };
            assert_eq!(
                word, expected,
                "{context}: original extra {extra} pixel {pixel}: {word:08x} != {expected:08x}"
            );
        }
    }
}

fn assert_same_words(actual: &Npy, baseline: &Npy, context: &str) {
    assert_eq!(actual.shape, baseline.shape, "{context}: off/on shape");
    assert_eq!(actual.words.len(), baseline.words.len());
    for (index, (&a, &b)) in actual.words.iter().zip(&baseline.words).enumerate() {
        assert_eq!(a, b, "{context}: off/on word {index}: {a:08x} != {b:08x}");
    }
}

#[test]
fn float_patch_fixtures_are_active_and_keep_special_words_outside_rectangles() {
    for case in cases() {
        let fixture = fixture(case);
        fixture.assert_active();
        assert!(fixture.expected_main.contains(&0x8000_0000));
        assert!(
            fixture
                .expected_main
                .iter()
                .any(|&w| w & 0x7f80_0000 == 0x7f80_0000 && w & 0x007f_ffff != 0)
        );
    }
}

#[test]
fn unsafe_float_patch_candidates_return_errors_without_publishing_output() {
    for (kind, message) in [
        (0, "reference is not representable"),
        (1, "residual is not representable"),
        (2, "not bit-exact"),
        (3, "not bit-exact"),
    ] {
        let mut words = vec![0x3c00u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                if glyph(x, y) {
                    words[y * WIDTH + x] = match kind {
                        0 => 0x0c00,                   // 2^-12 - 1 is not exactly F16.
                        1 if x >= WIDTH / 2 => 0x1000, // 1 + 2^-11 is not F16.
                        _ => 0,
                    };
                }
            }
        }
        if kind >= 2 {
            words[ORIGINS[1].1 * WIDTH + ORIGINS[1].0] = if kind == 2 { 0x8000 } else { 1 };
        }
        let colors = [words.iter().map(|&w| i32::from(w)).collect()];
        let config = SourcePatchConfig {
            floating_point: true,
            ..Default::default()
        };
        let error = patches::discover_source_float(
            FloatImageView {
                width: WIDTH,
                height: HEIGHT,
                format: SampleFormat::BINARY16,
                planes: &colors,
            },
            &config,
        )
        .unwrap_err();
        assert!(
            matches!(&error, Error::Encoding(_)) && error.to_string().contains(message),
            "case={kind}: {error}"
        );
        let image = Image::new(
            WIDTH as u32,
            HEIGHT as u32,
            PixelLayout::Gray,
            PixelData::F16(words),
        )
        .unwrap();
        let mut options = options(false);
        assert!(
            !encode(&image, &options).unwrap().is_empty(),
            "unpatched input must remain encodable"
        );
        options.source_patches = Some(config);
        let mut sink = vec![19, 23, 29];
        let error = encode_to(&image, &options, &mut sink).unwrap_err();
        assert!(
            matches!(&error, Error::Encoding(_)) && error.to_string().contains(message),
            "case={kind}: {error}"
        );
        assert_eq!(
            sink,
            [19, 23, 29],
            "failed Add must not silently fallback or publish a partial stream"
        );
    }
}

#[test]
fn independent_float_widening_and_npy_parser_preserve_special_bits() {
    for (word, expected) in [
        (0, 0),
        (0x8000, 0x8000_0000),
        (1, 0x3380_0000),
        (0x3c00, 0x3f80_0000),
        (0x7d31, 0x7fa6_2000),
        (0xfe23, 0xffc4_6000),
    ] {
        assert_eq!(widened_word(word, 16, 5), expected);
    }
    assert_eq!(widened_word(31 << 9, 16, 6), 1.0f32.to_bits());
    assert_eq!(widened_word(15 << 4, 10, 5), 1.0f32.to_bits());
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 1, 3), }\n";
    let words: [u32; 3] = [0x8000_0000, 0x7fa6_2000, 0xffc4_6000];
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    assert_eq!(
        parse_npy(&bytes).unwrap(),
        Npy {
            shape: [1, 1, 1, 3],
            words: words.to_vec()
        }
    );
    for end in 0..bytes.len() {
        assert!(parse_npy(&bytes[..end]).is_err());
    }
    bytes.push(0);
    assert!(parse_npy(&bytes).is_err());
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL official decoder path list"]
fn source_float_add_matches_original_bits_and_patches_off_official_decodes() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for decoder in &decoders {
        let banner = run_checked(
            Command::new(decoder).arg("--version"),
            &scratch.0,
            "djxl version",
        );
        assert!(
            banner.contains("djxl v"),
            "not an official djxl banner: {banner}"
        );
        eprintln!("{}\n{banner}", decoder.display());
    }
    let mut checks = 0;
    for (index, case) in cases().into_iter().enumerate() {
        let fixture = fixture(case);
        fixture.assert_active();
        let baseline_options = options(!index.is_multiple_of(2));
        let baseline = encode(&fixture.image, &baseline_options).unwrap();
        let mut enabled = baseline_options;
        enabled.source_patches = Some(fixture.config.clone());
        let patched = encode(&fixture.image, &enabled)
            .unwrap_or_else(|e| panic!("{case:?}: encode SourceAdd: {e}"));
        assert_ne!(
            patched, baseline,
            "{case:?}: SourceAdd must not silently encode an unpatched frame"
        );
        let direct_path = scratch.0.join(format!("case-{index}-off.jxl"));
        let patched_path = scratch.0.join(format!("case-{index}-on.jxl"));
        fs::write(&direct_path, baseline).unwrap();
        fs::write(&patched_path, patched).unwrap();
        for decoder in &decoders {
            let context = format!(
                "{case:?}/shared={}/{}",
                enabled.shared_tree,
                decoder.display()
            );
            let direct = decode(decoder, &direct_path, &scratch, &format!("{context}/off"));
            let actual = decode(decoder, &patched_path, &scratch, &format!("{context}/on"));
            assert_original(&direct, &fixture, &format!("{context}/off"));
            assert_original(&actual, &fixture, &format!("{context}/on"));
            assert_same_words(&actual, &direct, &context);
            checks += 1;
        }
    }
    eprintln!(
        "{checks} official float SourceAdd/off comparisons passed, including original float words and all extras"
    );
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
                "rs-jxl-float-patch-{}-{timestamp}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        }
        panic!("could not create a unique float-patch oracle directory");
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

fn run_checked(command: &mut Command, scratch: &Path, context: &str) -> String {
    let stdout = scratch.join("stdout.log");
    let stderr = scratch.join("stderr.log");
    // Redirect to files so diagnostics cannot deadlock on a full pipe.
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    let result = wait_with_timeout(command);
    command.stdout(Stdio::null()).stderr(Stdio::null()); // Release Windows handles.
    let diagnostics = format!(
        "stdout:\n{}\nstderr:\n{}",
        fs::read_to_string(stdout).unwrap_or_default(),
        fs::read_to_string(stderr).unwrap_or_default()
    );
    match result {
        Ok(Some(status)) if status.success() => diagnostics,
        Ok(Some(status)) => panic!("{context}: djxl exited {status}\n{diagnostics}"),
        Ok(None) => {
            panic!("{context}: djxl exceeded {COMMAND_TIMEOUT:?}; killed and reaped\n{diagnostics}")
        }
        Err(error) => panic!("{context}: launch/poll djxl: {error}\n{diagnostics}"),
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
                .unwrap_or_else(|e| panic!("resolve {}: {e}", path.display()))
        })
        .collect()
}

fn decode(decoder: &Path, input: &Path, scratch: &ScratchDir, context: &str) -> Npy {
    let output = scratch.0.join("output.npy");
    assert!(!output.exists(), "stale decoder output");
    // NPY includes all extras. Do not request color conversion, alpha
    // compositing, integer output, or separate extra-channel files.
    run_checked(
        Command::new(decoder)
            .arg(input)
            .arg(&output)
            .arg("--output_frames")
            .arg("--num_threads=1")
            .arg("--quiet"),
        &scratch.0,
        context,
    );
    let bytes = fs::read(&output).unwrap_or_else(|e| panic!("{context}: read NPY: {e}"));
    let result = parse_npy(&bytes).unwrap_or_else(|e| panic!("{context}: {e}"));
    fs::remove_file(output).unwrap();
    result
}
