// Official djxl is a development-only decoder oracle. Helpers follow reference.rs.
// Parent runs centrally with RS_JXL_DJXL (OS path list, ';' on Windows):
// cargo test -p rs-jxl --test palette_tools_reference -- --ignored --nocapture
// This bounded suite makes 23 encodes, checked by every supplied official decoder.

use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::modular::{
    group::{self, Channel, Transform},
    palette::{
        PaletteGeometry, PaletteMapping, PaletteMode, PaletteOptions, PaletteOrder, PaletteStage,
        PaletteState, PaletteTools, apply_palettes,
    },
    predictor::Predictor,
    transforms,
};
use rs_jxl::scheduler::{
    native::{NativeBackendOptions, encode_native_candidate},
    params::{CandidateParams, DeltaPalette, Override},
};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode};

const TIMEOUT: Duration = Duration::from_secs(90);
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Self {
        let root = env::temp_dir()
            .canonicalize()
            .expect("resolve temporary directory");
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        for _ in 0..100 {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "rs-jxl-palette-oracle-{}-{stamp}-{id}",
                std::process::id()
            ));
            assert_eq!(path.parent(), Some(root.as_path()));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create {}: {e}", path.display()),
            }
        }
        panic!("cannot create unique palette-oracle directory");
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Err(e) = fs::remove_dir_all(&self.0) {
            if thread::panicking() {
                eprintln!("cleanup {}: {e}", self.0.display());
            } else {
                panic!("cleanup {}: {e}", self.0.display());
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
        if start.elapsed() >= TIMEOUT {
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
    // Files avoid pipe backpressure; release handles before Windows cleanup.
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    let status = wait_with_timeout(command);
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let log = format!(
        "stdout:\n{}\nstderr:\n{}",
        fs::read_to_string(stdout).unwrap_or_default(),
        fs::read_to_string(stderr).unwrap_or_default()
    );
    match status {
        Ok(Some(status)) if status.success() => log,
        Ok(Some(status)) => panic!("{context}: djxl exited {status}\n{log}"),
        Ok(None) => panic!("{context}: djxl exceeded {TIMEOUT:?}; killed and reaped\n{log}"),
        Err(e) => panic!("{context}: launch/wait djxl: {e}\n{log}"),
    }
}

struct Oracle {
    decoders: Vec<PathBuf>,
    scratch: ScratchDir,
}

impl Oracle {
    fn new() -> Self {
        let value = env::var_os("RS_JXL_DJXL").expect(
            "explicit palette oracle test requires RS_JXL_DJXL: an OS path list of official djxl executables",
        );
        let decoders: Vec<_> = env::split_paths(&value)
            .map(|path| {
                assert!(
                    !path.as_os_str().is_empty() && path.is_file(),
                    "invalid official djxl path: {}",
                    path.display()
                );
                path.canonicalize().expect("resolve official djxl path")
            })
            .collect();
        assert!(
            !decoders.is_empty(),
            "RS_JXL_DJXL contains no decoder paths"
        );
        let scratch = ScratchDir::new();
        for decoder in &decoders {
            let banner = run_checked(
                Command::new(decoder).arg("--version"),
                &scratch.0,
                "decoder version",
            );
            assert!(
                banner.contains("djxl v"),
                "not an official djxl banner: {banner}"
            );
            eprintln!("{}: {}", decoder.display(), banner.trim());
        }
        Self { decoders, scratch }
    }

    fn check(&self, bytes: &[u8], image: &Image, context: &str) {
        assert!(!bytes.is_empty(), "{context}: empty codestream");
        let input = self.scratch.0.join("palette.jxl");
        fs::write(&input, bytes).unwrap();
        for decoder in &self.decoders {
            let output = self.scratch.0.join("palette.png");
            assert!(!output.exists(), "{context}: stale decoder output");
            let label = format!("{context}; decoder={}", decoder.display());
            run_checked(
                Command::new(decoder)
                    .arg(&input)
                    .arg(&output)
                    .arg(format!("--bits_per_sample={}", image.bits()))
                    .arg("--num_threads=1")
                    .arg("--quiet"),
                &self.scratch.0,
                &label,
            );
            assert_png_samples(&output, image, &label);
            fs::remove_file(output).unwrap();
        }
    }
}

fn assert_png_samples(path: &Path, image: &Image, context: &str) {
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path).unwrap()));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    let color = match image.layout() {
        PixelLayout::Rgb => png::ColorType::Rgb,
        PixelLayout::Rgba => png::ColorType::Rgba,
        _ => panic!("unexpected oracle layout"),
    };
    let depth = match image.bits() {
        8 => png::BitDepth::Eight,
        16 => png::BitDepth::Sixteen,
        _ => panic!("unexpected oracle depth"),
    };
    let info = reader.info();
    assert_eq!(
        (info.width, info.height),
        (image.width(), image.height()),
        "{context}"
    );
    assert_eq!(
        (info.color_type, info.bit_depth),
        (color, depth),
        "{context}"
    );
    assert!(
        info.animation_control.is_none(),
        "{context}: unexpected animation"
    );
    let expected: Vec<u16> = match image.pixels() {
        PixelData::U8(pixels) => pixels.iter().copied().map(u16::from).collect(),
        PixelData::U16(pixels) => pixels.clone(),
        _ => panic!("oracle requires U8/U16 input"),
    };
    let size = expected.len() * usize::from(image.bits() / 8);
    assert_eq!(reader.output_buffer_size(), Some(size), "{context}");
    let mut bytes = vec![0; size];
    let output = reader.next_frame(&mut bytes).unwrap();
    assert_eq!(output.buffer_size(), size, "{context}");
    assert_eq!(
        (output.color_type, output.bit_depth),
        (color, depth),
        "{context}"
    );
    let actual: Vec<u16> = if image.bits() == 16 {
        bytes
            .chunks_exact(2)
            .map(|v| u16::from_be_bytes([v[0], v[1]]))
            .collect()
    } else {
        bytes.into_iter().map(u16::from).collect()
    };
    assert_eq!(actual.len(), expected.len(), "{context}");
    // Compare every component, even when alpha is zero; no compositing oracle.
    if let Some((i, (a, b))) = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        let pixel = i / image.channel_count();
        panic!(
            "{context}: ({}, {}) channel {} expected {b}, got {a}",
            pixel % image.width() as usize,
            pixel / image.width() as usize,
            i % image.channel_count()
        );
    }
}

#[derive(Clone, Copy)]
enum Pattern {
    Palette,
    Local,
    DeltaFallback,
}

fn fixture(width: u32, height: u32, bits: u8, layout: PixelLayout, pattern: Pattern) -> Image {
    let colors: [[u16; 4]; 8] = [
        [0, 0, 0, 0],
        [255, 255, 255, 255],
        [1, 2, 3, 0],
        [7, 11, 13, 0],
        [7, 11, 13, 255],
        [90, 10, 120, 127],
        [32, 32, 32, 0],
        [254, 1, 200, 255],
    ];
    let count = if layout == PixelLayout::Rgba { 4 } else { 3 };
    let scale = if bits == 16 { 257 } else { 1 };
    let mut samples = Vec::new();
    for pos in 0..width as usize * height as usize {
        let x = pos % width as usize;
        let y = pos / width as usize;
        let color = match pattern {
            Pattern::Palette => {
                let slot = (pos + y * 5) % 64;
                let id = if pos < 8 {
                    pos
                } else {
                    match slot {
                        0..=31 => 0,
                        32..=47 => 2,
                        48..=55 => 3,
                        56..=59 => 4,
                        60..=61 => 5,
                        62 => 6,
                        _ => 1,
                    }
                };
                colors[id]
            }
            Pattern::Local => {
                let group = (x / 128) as u16;
                let i = ((x / 4 + y * 3) % 8) as u16;
                [
                    10 + group * 100 + i * 3,
                    20 + i * 5,
                    30 + group * 80 + i * 7,
                    if i.is_multiple_of(3) { 0 } else { 255 },
                ]
            }
            Pattern::DeltaFallback => {
                let colors = [[10, 20, 30, 0], [50, 60, 70, 255], [110, 140, 170, 0]];
                colors[[0, 0, 1, 0, 2, 1, 2, 2][x % 8]]
            }
        };
        samples.extend(color[..count].iter().map(|&value| {
            let scaled = value * scale;
            // Keep exact black/white/transparent endpoints, but use real U16
            // low bits so an accidental 8-bit roundtrip cannot pass the oracle.
            if bits == 16 && (1..255).contains(&value) {
                scaled + value % 7
            } else {
                scaled
            }
        }));
    }
    let pixels = if bits == 16 {
        PixelData::U16(samples)
    } else {
        PixelData::U8(
            samples
                .into_iter()
                .map(|v| u8::try_from(v).unwrap())
                .collect(),
        )
    };
    Image::new(width, height, layout, pixels).unwrap()
}

fn tools(order: u8, geometry: u8, mapping: PaletteMapping) -> PaletteTools {
    PaletteTools {
        order: PaletteOrder::from_source_mode(order).unwrap(),
        geometry: PaletteGeometry::from_source_mode(geometry).unwrap(),
        mapping,
    }
}

fn options(tools: PaletteTools, limit: i32, mode: PaletteMode) -> EncodeOptions {
    EncodeOptions {
        group_size_shift: 0,
        modular: ModularOptions {
            predictor: Predictor::Gradient,
            rct: None,
            palette_policy: Some(PaletteOptions {
                palette_colors: limit,
                mode,
                tools,
                channel_colors_pre_transform_percent: 0.0,
                channel_colors_percent: 0.0,
                ..PaletteOptions::default()
            }),
            ..ModularOptions::default()
        },
        ..EncodeOptions::default()
    }
}

fn channels(image: &Image, x0: usize, width: usize) -> Vec<Channel> {
    (0..image.channel_count())
        .map(|c| Channel {
            width,
            height: image.height() as usize,
            hshift: 0,
            vshift: 0,
            samples: (0..image.height() as usize)
                .flat_map(|y| {
                    (x0..x0 + width).map(move |x| {
                        let index = (y * image.width() as usize + x) * image.channel_count() + c;
                        match image.pixels() {
                            PixelData::U8(p) => i32::from(p[index]),
                            PixelData::U16(p) => i32::from(p[index]),
                            _ => panic!("expected unsigned integer fixture"),
                        }
                    })
                })
                .collect(),
        })
        .collect()
}

fn preflight_global(image: &Image, options: &EncodeOptions, deltas: usize) {
    let mut channels = channels(image, 0, image.width() as usize);
    let mut state = PaletteState::new(image.bits()).unwrap();
    let script = apply_palettes(
        &mut channels,
        &mut state,
        image.bits(),
        PaletteStage::PreRct,
        options.modular.palette_policy.as_ref().unwrap(),
    )
    .unwrap();
    assert_eq!(script.len(), 1, "expected one global palette");
    assert!(
        matches!(script[0], Transform::Palette { channels: count, deltas: d, .. }
        if count == image.channel_count() && d == deltas),
        "{script:?}"
    );
    assert_eq!(state.meta_channels, 1);
}

fn preflight_local(image: &Image, options: &EncodeOptions) {
    assert_eq!(image.width(), 256);
    assert_eq!(options.group_size_shift, 0);
    assert_eq!(options.modular.rct, Some(6));
    let policy = options.modular.palette_policy.as_ref().unwrap();
    let mut full = channels(image, 0, image.width() as usize);
    let before = full.clone();
    let mut state = PaletteState::new(image.bits()).unwrap();
    assert!(
        apply_palettes(
            &mut full,
            &mut state,
            image.bits(),
            PaletteStage::PreRct,
            policy
        )
        .unwrap()
        .is_empty()
    );
    assert_eq!(full, before, "global candidate must decline unchanged");
    for x0 in [0, 128] {
        let mut local = channels(image, x0, 128);
        assert!(group::apply_rct(&mut local, 0, 0, 6).unwrap().is_some());
        assert!(
            local.iter().any(|c| c.samples.iter().any(|&v| v < 0)),
            "exercise signed post-RCT tuples"
        );
        let mut state = PaletteState {
            meta_channels: 0,
            max_bitdepth: 0,
            max_value: 0,
        };
        let script = apply_palettes(
            &mut local,
            &mut state,
            image.bits(),
            PaletteStage::PostRct,
            policy,
        )
        .unwrap();
        assert!(
            matches!(
                &script[..],
                [Transform::Palette {
                    channels: 4,
                    colors: 8,
                    deltas: 0,
                    ..
                }]
            ),
            "{script:?}"
        );
    }
}

fn preflight_delta_fallback(image: &Image, options: &EncodeOptions) {
    let planes: Vec<_> = channels(image, 0, image.width() as usize)
        .into_iter()
        .map(|c| c.samples)
        .collect();
    assert!(
        transforms::delta_palette_forward(
            &planes,
            0,
            4,
            image.width() as usize,
            image.height() as usize,
            3,
            u32::from(image.bits()),
            Predictor::Left
        )
        .unwrap()
        .is_none(),
        "fixture must exceed residual palette limit"
    );
    preflight_global(image, options, 0);
}

#[test]
fn oracle_fixtures_activate_global_local_and_delta_fallback_routes() {
    let selected = tools(9, 0, PaletteMapping::Flat);
    let global = fixture(257, 9, 8, PixelLayout::Rgba, Pattern::Palette);
    let PixelData::U8(pixels) = global.pixels() else {
        panic!("U8 fixture");
    };
    assert!(
        pixels
            .chunks_exact(4)
            .any(|p| p[3] == 0 && p[..3].iter().any(|&v| v != 0))
    );
    preflight_global(&global, &options(selected, 16, PaletteMode::Absolute), 0);
    let local = fixture(256, 8, 16, PixelLayout::Rgba, Pattern::Local);
    let PixelData::U16(pixels) = local.pixels() else {
        panic!("U16 fixture");
    };
    assert!(
        pixels.iter().any(|v| v % 257 != 0),
        "exercise genuine U16 precision"
    );
    let mut local_options = options(selected, 8, PaletteMode::Absolute);
    local_options.modular.rct = Some(6);
    preflight_local(&local, &local_options);
    let delta = fixture(32, 8, 16, PixelLayout::Rgba, Pattern::DeltaFallback);
    preflight_delta_fallback(
        &delta,
        &options(selected, 3, PaletteMode::ExactDelta(Predictor::Left)),
    );
    preflight_global(
        &delta,
        &options(selected, 3, PaletteMode::ExactZero(Predictor::Left)),
        1,
    );
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL official decoder path list"]
fn global_orders_zero_through_nine_preserve_hidden_rgba() {
    let oracle = Oracle::new();
    let image = fixture(257, 9, 8, PixelLayout::Rgba, Pattern::Palette);
    let cases = (0..=9)
        .map(|order| {
            (
                order,
                if order % 2 == 0 {
                    PaletteMapping::Tree
                } else {
                    PaletteMapping::Flat
                },
            )
        })
        .chain([(0, PaletteMapping::Flat), (9, PaletteMapping::Tree)]);
    for (order, mapping) in cases {
        let mut options = options(tools(order, 0, mapping), 16, PaletteMode::Absolute);
        options.shared_tree = order % 2 == 1;
        options.modular.palette_policy.as_mut().unwrap().direct = order == 5;
        preflight_global(&image, &options, 0);
        let label = format!(
            "global order={order} geometry=0 map={mapping:?} shared={}",
            options.shared_tree
        );
        oracle.check(&encode(&image, &options).unwrap(), &image, &label);
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL official decoder path list"]
fn principal_geometry_overrides_order_and_preserves_u16_samples() {
    let oracle = Oracle::new();
    for geometry in [1, 2] {
        for mapping in [PaletteMapping::Tree, PaletteMapping::Flat] {
            let layout = if geometry == 1 {
                PixelLayout::Rgb
            } else {
                PixelLayout::Rgba
            };
            let image = fixture(32, 16, 16, layout, Pattern::Palette);
            let options = options(tools(9, geometry, mapping), 16, PaletteMode::Absolute);
            preflight_global(&image, &options, 0);
            let planes: Vec<_> = channels(&image, 0, 32)
                .into_iter()
                .map(|c| c.samples)
                .collect();
            let selected = transforms::palette_forward_with_options(
                &planes,
                0,
                image.channel_count(),
                32,
                16,
                16,
                16,
                true,
                tools(9, geometry, mapping),
            )
            .unwrap()
            .unwrap();
            let geometry_only = transforms::palette_forward_with_options(
                &planes,
                0,
                image.channel_count(),
                32,
                16,
                16,
                16,
                true,
                tools(0, geometry, mapping),
            )
            .unwrap()
            .unwrap();
            assert_eq!(selected, geometry_only);
            oracle.check(
                &encode(&image, &options).unwrap(),
                &image,
                &format!("U16 {layout:?} order=9 geometry={geometry} map={mapping:?}"),
            );
        }
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL official decoder path list"]
fn local_palette_after_global_decline_and_ycocg_preserves_samples() {
    let oracle = Oracle::new();
    for (bits, order, mapping) in [(8, 8, PaletteMapping::Flat), (16, 5, PaletteMapping::Tree)] {
        let image = fixture(256, 8, bits, PixelLayout::Rgba, Pattern::Local);
        let mut options = options(tools(order, 0, mapping), 8, PaletteMode::Absolute);
        options.modular.rct = Some(6);
        options.shared_tree = bits == 16;
        preflight_local(&image, &options);
        oracle.check(
            &encode(&image, &options).unwrap(),
            &image,
            &format!("local RGBA bits={bits} RCT=6 order={order} map={mapping:?}"),
        );
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL official decoder path list"]
fn delta_fallback_and_exact_zero_compose_with_optional_ordering() {
    let oracle = Oracle::new();
    for (bits, order, mapping, mode, deltas) in [
        (
            8,
            1,
            PaletteMapping::Tree,
            PaletteMode::ExactDelta(Predictor::Left),
            0,
        ),
        (
            16,
            6,
            PaletteMapping::Flat,
            PaletteMode::ExactDelta(Predictor::Left),
            0,
        ),
        (
            16,
            7,
            PaletteMapping::Flat,
            PaletteMode::ExactZero(Predictor::Left),
            1,
        ),
    ] {
        let image = fixture(32, 8, bits, PixelLayout::Rgba, Pattern::DeltaFallback);
        let options = options(tools(order, 0, mapping), 3, mode);
        if deltas == 0 {
            preflight_delta_fallback(&image, &options);
        }
        preflight_global(&image, &options, deltas);
        oracle.check(
            &encode(&image, &options).unwrap(),
            &image,
            &format!("RGBA bits={bits} order={order} map={mapping:?} mode={mode:?}"),
        );
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL official decoder path list"]
fn native_source_adapter_retains_explicit_palette_tools_in_encoded_artifact() {
    let oracle = Oracle::new();
    for (bits, selected, delta) in [
        (
            16,
            tools(6, 0, PaletteMapping::Flat),
            DeltaPalette::Disabled,
        ),
        (8, tools(9, 2, PaletteMapping::Tree), DeltaPalette::ZeroLeft),
    ] {
        let image = fixture(32, 16, bits, PixelLayout::Rgba, Pattern::Palette);
        // These explicit frozen-source extensions are ambient base policy, not
        // CandidateParams defaults. Other policy fields must be rebuilt.
        let base = options(selected, 777, PaletteMode::Absolute);
        let params = CandidateParams {
            palette_colors: 16,
            rct: Some(0),
            predictor: Some(Predictor::Gradient),
            group_size_shift: Some(0),
            patches: Override::Off,
            delta_palette: delta,
            channel_colors_pre_transform_percent: 0.0,
            channel_colors_percent: 0.0,
            ..CandidateParams::default()
        };
        let result =
            encode_native_candidate(&params, &image, &base, NativeBackendOptions::default())
                .unwrap();
        assert!(
            result.artifact.gaps.is_empty(),
            "{:?}",
            result.artifact.gaps
        );
        assert_eq!(result.artifact.effective_params, params);
        let policy = result
            .artifact
            .options
            .modular
            .palette_policy
            .as_ref()
            .unwrap();
        assert_eq!(
            policy.tools, selected,
            "adapter discarded explicit palette tools"
        );
        assert_eq!(policy.palette_colors, 16);
        assert_eq!(
            policy.mode,
            if delta == DeltaPalette::Disabled {
                PaletteMode::Absolute
            } else {
                PaletteMode::ExactZero(Predictor::Left)
            }
        );
        assert_eq!(
            base.modular.palette_policy.as_ref().unwrap().tools,
            selected
        );
        assert_eq!(
            base.modular.palette_policy.as_ref().unwrap().palette_colors,
            777
        );
        preflight_global(
            &image,
            &result.artifact.options,
            usize::from(delta != DeltaPalette::Disabled),
        );
        oracle.check(
            &result.bytes,
            &image,
            &format!("native adapter RGBA bits={bits} tools={selected:?} delta={delta:?}"),
        );
    }
}
