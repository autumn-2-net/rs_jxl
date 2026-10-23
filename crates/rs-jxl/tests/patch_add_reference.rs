// Source Add patches: fixture activation is mandatory, not a fallback pass.
// Uses existing png dev-dependency and explicitly supplied official decoders.
// RS_JXL_DJXL is an OS path list (semicolon-separated on Windows), preferably
// containing both official djxl 0.12 and 0.13. Parent runs, not this agent:
// cargo test -p rs-jxl --test patch_add_reference -- --ignored --nocapture

use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::metadata::{ExtraChannelInfo, ExtraChannelType};
use rs_jxl::tools::patches::{self, AddPatchPlan, ImageView, PatchBlendMode, SourcePatchConfig};
use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout, encode};

const TIMEOUT: Duration = Duration::from_secs(30);
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

// Kept independent of reference.rs: no module inclusion or edits to its tests.
struct ScratchDir {
    root: PathBuf,
    path: PathBuf,
}

impl ScratchDir {
    fn new() -> Self {
        let root = env::temp_dir().canonicalize().expect("resolve temp root");
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        for _ in 0..100 {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "rs-jxl-patch-add-{}-{stamp}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self { root, path },
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create {}: {e}", path.display()),
            }
        }
        panic!("cannot create unique patch Add scratch directory");
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        // Only remove the resolved direct child created by this instance.
        let valid = self.path.parent() == Some(self.root.as_path())
            && self.path.canonicalize().is_ok_and(|path| path == self.path);
        if !valid {
            eprintln!(
                "refusing cleanup of changed scratch path {}",
                self.path.display()
            );
            return;
        }
        if let Err(e) = fs::remove_dir_all(&self.path) {
            if thread::panicking() {
                eprintln!("cleanup {}: {e}", self.path.display());
            } else {
                panic!("cleanup {}: {e}", self.path.display());
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

fn wait(command: &mut Command) -> io::Result<Option<ExitStatus>> {
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

fn run(command: &mut Command, scratch: &Path, context: &str) -> String {
    let stdout = scratch.join("stdout.log");
    let stderr = scratch.join("stderr.log");
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    let status = wait(command);
    // Release Windows redirection handles before directory cleanup.
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let log = format!(
        "stdout:\n{}\nstderr:\n{}",
        fs::read_to_string(stdout).unwrap_or_default(),
        fs::read_to_string(stderr).unwrap_or_default()
    );
    match status {
        Ok(Some(status)) if status.success() => log,
        Ok(Some(status)) => panic!("{context}: djxl exited {status}\n{log}"),
        Ok(None) => panic!("{context}: djxl exceeded {TIMEOUT:?}, killed and reaped\n{log}"),
        Err(e) => panic!("{context}: launch djxl: {e}\n{log}"),
    }
}

fn decoders() -> Vec<PathBuf> {
    let setting = env::var_os("RS_JXL_DJXL").expect(
        "explicit official Add test requires RS_JXL_DJXL path list; missing oracles are not a skip",
    );
    let paths: Vec<_> = env::split_paths(&setting)
        .map(|path| {
            assert!(
                !path.as_os_str().is_empty() && path.is_file(),
                "invalid djxl path {}",
                path.display()
            );
            path.canonicalize().expect("resolve djxl executable")
        })
        .collect();
    assert!(!paths.is_empty(), "RS_JXL_DJXL has no decoder paths");
    paths
}

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Backgrounds,
    CoarseFingerprint,
    AboveNominalResidual,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    width: usize,
    bits: u32,
    layout: PixelLayout,
    pattern: Pattern,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for bits in [8, 16] {
        for layout in [
            PixelLayout::Gray,
            PixelLayout::GrayAlpha,
            PixelLayout::Rgb,
            PixelLayout::Rgba,
        ] {
            for width in [129, 385] {
                cases.push(Case {
                    width,
                    bits,
                    layout,
                    pattern: Pattern::Backgrounds,
                });
            }
        }
        for layout in [PixelLayout::Gray, PixelLayout::Rgba] {
            for pattern in [Pattern::CoarseFingerprint, Pattern::AboveNominalResidual] {
                cases.push(Case {
                    width: 385,
                    bits,
                    layout,
                    pattern,
                });
            }
        }
    }
    cases
}

fn pixels(values: Vec<u16>, bits: u32) -> PixelData {
    if bits == 8 {
        PixelData::U8(
            values
                .into_iter()
                .map(|v| u8::try_from(v).unwrap())
                .collect(),
        )
    } else {
        PixelData::U16(values)
    }
}

struct Fixture {
    image: Image,
    planes: Vec<Vec<i32>>,
    config: SourcePatchConfig,
    origins: [(usize, usize); 4],
    case: Case,
}

impl Fixture {
    fn view(&self) -> ImageView<'_> {
        ImageView {
            width: self.image.width() as usize,
            height: self.image.height() as usize,
            bit_depth: self.case.bits,
            color_channels: self.image.color_channels(),
            planes: &self.planes,
        }
    }

    fn assert_active(&self) -> AddPatchPlan {
        let plan = patches::discover_source(self.view(), &self.config)
            .unwrap()
            .unwrap_or_else(|| {
                panic!(
                    "{:?}: fixture did not activate source Add patches",
                    self.case
                )
            });
        assert!(!plan.dictionary.references.is_empty());
        assert!(
            plan.dictionary.positions.len() >= 4,
            "{:?}: repeated glyphs were not grouped",
            self.case
        );
        // The source key must share one glyph across the separate backgrounds
        // or across deliberately different samples in the coarsened-key case.
        assert!(
            plan.dictionary
                .references
                .iter()
                .enumerate()
                .any(|(index, _)| {
                    self.origins.iter().all(|&(x, y)| {
                        plan.dictionary.positions.iter().any(|p| {
                            p.ref_pos_idx == index && p.x as usize == x && p.y as usize == y
                        })
                    })
                }),
            "{:?}: four glyphs must share an atlas entry",
            self.case
        );
        assert!(plan.requires_signed_samples());
        match self.case.pattern {
            Pattern::Backgrounds => assert!(
                plan.reference.planes[..self.image.color_channels()]
                    .iter()
                    .flatten()
                    .any(|&v| v < 0)
            ),
            Pattern::CoarseFingerprint => assert!(
                plan.residual.planes[..self.image.color_channels()]
                    .iter()
                    .flatten()
                    .any(|&v| v < 0)
            ),
            Pattern::AboveNominalResidual => assert!(
                plan.residual.planes[..self.image.color_channels()]
                    .iter()
                    .flatten()
                    .any(|&v| v > (1i32 << self.case.bits) - 1)
            ),
        }
        let stride = 1 + plan.dictionary.num_extra_channels;
        for blends in plan.dictionary.blendings.chunks_exact(stride) {
            assert_eq!(blends[0].mode, PatchBlendMode::Add);
            assert!(blends[1..].iter().all(|b| b.mode == PatchBlendMode::None));
        }
        assert_eq!(
            patches::compose_add(&plan.residual, &plan.reference, &plan.dictionary).unwrap(),
            self.planes
        );
        assert_eq!(
            &plan.residual.planes[self.image.color_channels()..],
            &self.planes[self.image.color_channels()..]
        );
        if self.image.layout().has_alpha() {
            let alpha = self.image.color_channels();
            for &(x, y) in &self.origins {
                let at = y * self.image.width() as usize + x;
                assert_eq!(self.planes[alpha][at], 0);
                assert!(
                    self.planes[..alpha].iter().all(|p| p[at] != 0),
                    "hidden colors must not all be zero"
                );
            }
        }
        plan
    }
}

fn fixture(case: Case) -> Fixture {
    let width = case.width;
    let height = 83;
    let channels = case.layout.channels();
    let colors = if channels < 3 { 1 } else { 3 };
    let extra_count = if case.layout == PixelLayout::Rgba {
        3
    } else {
        0
    };
    let maximum = (1i32 << case.bits) - 1;
    let scale = if case.bits == 16 { 257 } else { 1 };
    let origins = [
        (17, 17),
        (width / 2 + 17, 17),
        (33, 53),
        (width / 2 + 33, 53),
    ];
    let mut planes = vec![vec![0; width * height]; channels + extra_count];
    let mut interleaved = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        for x in 0..width {
            let glyph = origins.iter().enumerate().find_map(|(index, &(x0, y0))| {
                if x < x0 || y < y0 {
                    return None;
                }
                let (gx, gy) = (x - x0, y - y0);
                (gx < 7
                    && gy < 9
                    && (gx < 2 || !(2..7).contains(&gy) || ((4..6).contains(&gy) && gx < 6)))
                    .then_some((index, gx, gy))
            });
            let at = y * width + x;
            for (c, plane) in planes.iter_mut().enumerate().take(colors) {
                let background = match case.pattern {
                    Pattern::Backgrounds => {
                        ((if x < width / 2 { 185 } else { 235 }) - c as i32 * 23) * scale
                            + if case.bits == 16 {
                                7 + c as i32 * 13
                            } else {
                                0
                            }
                    }
                    Pattern::CoarseFingerprint => 0,
                    Pattern::AboveNominalResidual => maximum,
                };
                let value = if let Some((index, gx, gy)) = glyph {
                    match case.pattern {
                        Pattern::Backgrounds => {
                            background - (63 + c as i32 * 14 - ((gx + gy) % 3) as i32) * scale
                                + if case.bits == 16 {
                                    ((gx * 3 + gy) % 19) as i32
                                } else {
                                    0
                                }
                        }
                        Pattern::CoarseFingerprint => {
                            (if index == 0 || index == 2 { 64 } else { 44 }) * scale
                        }
                        Pattern::AboveNominalResidual => {
                            maximum - (if index == 0 || index == 2 { 64 } else { 45 }) * scale
                        }
                    }
                } else {
                    background
                };
                assert!((0..=maximum).contains(&value));
                plane[at] = value;
            }
            if case.layout.has_alpha() {
                planes[colors][at] = if glyph.is_some() {
                    0
                } else {
                    match (x / 5 + y / 7) % 3 {
                        0 => 0,
                        1 => maximum,
                        _ => maximum / 2,
                    }
                };
            }
            for (extra, plane) in planes[channels..].iter_mut().enumerate() {
                plane[at] = ((at * 173 + extra * 977 + (x % 17) * 251) as i32) & maximum;
            }
            for plane in &planes[..channels] {
                interleaved.push(plane[at] as u16);
            }
        }
    }
    let image = Image::new(
        width as u32,
        height as u32,
        case.layout,
        pixels(interleaved, case.bits),
    )
    .unwrap();
    let mut metadata = image.metadata().clone();
    let extras: Vec<_> = planes[channels..]
        .iter()
        .enumerate()
        .map(|(i, plane)| {
            let kind = [
                ExtraChannelType::Depth,
                ExtraChannelType::SelectionMask,
                ExtraChannelType::Optional,
            ][i];
            let mut info = ExtraChannelInfo::new(kind, case.bits as u8).unwrap();
            info.name = format!("patch-add-{kind:?}");
            metadata.extra_channels.push(info);
            pixels(plane.iter().map(|&v| v as u16).collect(), case.bits)
        })
        .collect();
    let image = image.with_metadata(metadata, extras).unwrap();
    let config = SourcePatchConfig {
        fingerprint_divisor: if matches!(case.pattern, Pattern::Backgrounds) {
            1
        } else {
            2
        },
        extra_channel_dim_shifts: vec![0; image.metadata().extra_channels.len()],
        ..SourcePatchConfig::default()
    };
    Fixture {
        image,
        planes,
        config,
        origins,
        case,
    }
}

fn assert_png(path: &Path, image: &Image, context: &str) {
    let file =
        File::open(path).unwrap_or_else(|e| panic!("{context}: open {}: {e}", path.display()));
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    let color = match image.layout() {
        PixelLayout::Gray => png::ColorType::Grayscale,
        PixelLayout::GrayAlpha => png::ColorType::GrayscaleAlpha,
        PixelLayout::Rgb => png::ColorType::Rgb,
        PixelLayout::Rgba => png::ColorType::Rgba,
    };
    let depth = if image.bits() == 8 {
        png::BitDepth::Eight
    } else {
        png::BitDepth::Sixteen
    };
    assert_eq!(
        (reader.info().width, reader.info().height),
        (image.width(), image.height()),
        "{context}: dimensions"
    );
    assert_eq!(
        (reader.info().color_type, reader.info().bit_depth),
        (color, depth),
        "{context}: sample format"
    );
    assert!(
        reader.info().animation_control.is_none(),
        "{context}: reference-only atlas must not become a displayed frame"
    );
    let expected: Vec<_> = match image.pixels() {
        PixelData::U8(v) => v.iter().copied().map(u16::from).collect(),
        PixelData::U16(v) => v.clone(),
        _ => panic!("test only accepts U8/U16 PNG fixtures"),
    };
    let size = expected.len() * usize::from(image.bits() / 8);
    assert_eq!(
        reader.output_buffer_size(),
        Some(size),
        "{context}: sample count"
    );
    let mut bytes = vec![0; size];
    let output = reader.next_frame(&mut bytes).unwrap();
    assert_eq!(output.buffer_size(), size);
    assert_eq!((output.color_type, output.bit_depth), (color, depth));
    let actual: Vec<_> = if image.bits() == 8 {
        bytes.into_iter().map(u16::from).collect()
    } else {
        bytes
            .chunks_exact(2)
            .map(|v| u16::from_be_bytes([v[0], v[1]]))
            .collect()
    };
    assert_eq!(actual.len(), expected.len());
    if let Some((i, (actual, expected))) = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        let pixel = i / image.layout().channels();
        panic!(
            "{context}: ({},{}) channel {} expected {expected}, got {actual}",
            pixel % image.width() as usize,
            pixel / image.width() as usize,
            i % image.layout().channels()
        );
    }
}

fn extra_png(main: &Path, index: usize) -> PathBuf {
    let stem = main.file_stem().unwrap().to_str().unwrap();
    let name = main.file_name().unwrap().to_str().unwrap();
    // The official versions differ in whether the base extension is retained.
    let candidates = [
        main.with_file_name(format!("{stem}-ec{}.png", index + 1)),
        main.with_file_name(format!("{name}-ec{}.png", index + 1)),
    ];
    candidates
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| {
            panic!(
                "missing extra {} for {}; files: {:?}",
                index,
                main.display(),
                fs::read_dir(main.parent().unwrap())
                    .unwrap()
                    .map(|p| p.unwrap().file_name())
                    .collect::<Vec<_>>()
            )
        })
}

#[test]
fn source_add_oracle_fixtures_really_activate() {
    for case in cases() {
        let fixture = fixture(case);
        let fast = fixture.assert_active();
        let baseline = SourcePatchConfig {
            fast: false,
            ..fixture.config.clone()
        };
        let baseline = patches::discover_source(fixture.view(), &baseline)
            .unwrap()
            .unwrap();
        assert_eq!(
            fast.dictionary, baseline.dictionary,
            "{case:?}: fast/baseline dictionary"
        );
        assert_eq!(
            fast.reference, baseline.reference,
            "{case:?}: fast/baseline atlas"
        );
        assert_eq!(
            fast.residual, baseline.residual,
            "{case:?}: fast/baseline residual"
        );
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL official decoder path list"]
fn source_add_signed_atlas_and_residual_match_official_djxl() {
    let decoders = decoders();
    let scratch = ScratchDir::new();
    for decoder in &decoders {
        let banner = run(
            Command::new(decoder).arg("--version"),
            &scratch.path,
            "djxl version",
        );
        assert!(
            banner.contains("djxl v"),
            "not a djxl version banner: {banner}"
        );
        eprintln!("{}\n{banner}", decoder.display());
    }
    let mut main_checks = 0;
    let mut extra_checks = 0;
    for (case_index, case) in cases().into_iter().enumerate() {
        let fixture = fixture(case);
        fixture.assert_active();
        for shared_tree in [false, true] {
            let context = format!("source Add {case:?} shared={shared_tree}");
            let options = EncodeOptions {
                threads: 2,
                shared_tree,
                max_output_bytes: 16 << 20,
                patches: None,
                source_patches: Some(fixture.config.clone()),
                ..EncodeOptions::default()
            };
            let encoded =
                encode(&fixture.image, &options).unwrap_or_else(|e| panic!("{context}: {e}"));
            let mut disabled = options.clone();
            disabled.source_patches = None;
            let direct = encode(&fixture.image, &disabled).unwrap();
            assert_ne!(
                encoded, direct,
                "{context}: source route silently encoded the direct frame"
            );
            let input = scratch
                .path
                .join(format!("case-{case_index}-s{}.jxl", u8::from(shared_tree)));
            fs::write(&input, encoded).unwrap();
            for (decoder_index, decoder) in decoders.iter().enumerate() {
                let context = format!("{context}, decoder={}", decoder.display());
                let output = scratch.path.join(format!(
                    "case-{case_index}-s{}-d{decoder_index}.png",
                    u8::from(shared_tree)
                ));
                assert!(!output.exists(), "unique oracle output must not preexist");
                let mut command = Command::new(decoder);
                command
                    .arg(&input)
                    .arg(&output)
                    .arg(format!("--bits_per_sample={}", case.bits))
                    .arg("--num_threads=1")
                    .arg("--quiet");
                if !fixture.image.extra_planes().is_empty() {
                    command.arg("--output_extra_channels");
                }
                run(&mut command, &scratch.path, &context);
                assert_png(&output, &fixture.image, &context);
                main_checks += 1;
                for (index, data) in fixture.image.extra_planes().iter().enumerate() {
                    let expected = Image::new(
                        fixture.image.width(),
                        fixture.image.height(),
                        PixelLayout::Gray,
                        data.clone(),
                    )
                    .unwrap();
                    assert_png(
                        &extra_png(&output, index),
                        &expected,
                        &format!("{context}, extra={index}"),
                    );
                    extra_checks += 1;
                }
            }
        }
    }
    eprintln!(
        "{main_checks} official source-Add main-image checks and {extra_checks} extra-channel checks passed"
    );
}

fn projected_patch_case(
    speed_tier: i8,
    predictor: rs_jxl::modular::predictor::Predictor,
    layout: PixelLayout,
) -> (Fixture, EncodeOptions) {
    use rs_jxl::modular::predictor::Predictor;
    use rs_jxl::scheduler::adapter::{AdaptationState, project_native};
    use rs_jxl::scheduler::params::{CandidateParams, Override};

    let mut fixture = fixture(Case {
        width: 385,
        bits: 16,
        layout,
        pattern: Pattern::Backgrounds,
    });
    let params = CandidateParams {
        speed_tier,
        predictor: Some(predictor),
        patches: Override::Default,
        group_size_shift: None,
        ..CandidateParams::default()
    };
    let base = EncodeOptions {
        threads: 2,
        max_output_bytes: 16 << 20,
        ..EncodeOptions::default()
    };
    let projection = project_native(&params, &fixture.image, &base).unwrap();
    assert!(
        projection.gaps.is_empty(),
        "tier={speed_tier} predictor={predictor:?} layout={layout:?}: {:?}",
        projection.gaps
    );
    assert_eq!(projection.state(), AdaptationState::SourceEquivalent);
    let options = projection.exact_options().unwrap();
    assert!(
        options.patches.is_none(),
        "source projection must not select Replace"
    );
    assert!(options.shared_tree);
    fixture.config = options
        .source_patches
        .clone()
        .expect("automatic source Add policy");
    assert_eq!(fixture.config.speed_tier, speed_tier);
    let plan = fixture.assert_active();

    let policy = options
        .source_patch_reference
        .as_ref()
        .expect("deferred source reference policy");
    assert!(
        policy.group_size_shift.is_none(),
        "retain auto sentinel until atlas exists"
    );
    assert!(
        policy.modular.sampling.is_none(),
        "reference sampling is deferred"
    );
    assert_eq!(policy.modular.predictor, Predictor::Gradient);
    assert_eq!(policy.sampling.learning_fraction, params.learning_fraction);
    let (width, height) = (plan.reference.width as u32, plan.reference.height as u32);
    assert_ne!(
        (width, height),
        (fixture.image.width(), fixture.image.height())
    );
    // Metadata-only shape for public policy.resolve. This is NEVER encoded and
    // never casts signed atlas samples to U16; the backend uses signed planes.
    let count = width as usize * height as usize;
    let atlas_shape = Image::new(
        width,
        height,
        layout,
        PixelData::U16(vec![0; count * layout.channels()]),
    )
    .unwrap()
    .with_metadata(
        fixture.image.metadata().clone(),
        vec![PixelData::U16(vec![0; count]); fixture.image.extra_planes().len()],
    )
    .unwrap();
    let resolved = policy.resolve(&atlas_shape).unwrap();
    assert_eq!(options.group_size_shift, 2, "385-wide source auto geometry");
    assert_eq!(
        resolved.group_size_shift, 0,
        "small atlas must resolve its OWN geometry"
    );
    assert_ne!(options.group_size_shift, resolved.group_size_shift);

    // Independent reference projection starts from original candidate inputs,
    // with source RoundtripPatchFrame's Gradient/patches-Off overrides.
    let reference_params = CandidateParams {
        predictor: Some(Predictor::Gradient),
        patches: Override::Off,
        ..params
    };
    let reference_projection = project_native(&reference_params, &atlas_shape, &base).unwrap();
    assert!(
        reference_projection.gaps.is_empty(),
        "reference gaps: {:?}",
        reference_projection.gaps
    );
    let expected = reference_projection.exact_options().unwrap();
    assert_eq!(resolved.group_size_shift, expected.group_size_shift);
    assert_eq!(resolved.modular.predictor, expected.modular.predictor);
    assert_eq!(
        resolved.modular.weighted_candidates,
        expected.modular.weighted_candidates
    );
    assert_eq!(
        resolved.modular.source_lz77.as_ref().map(|c| c.method),
        expected.modular.source_lz77.as_ref().map(|c| c.method)
    );
    let sampling = resolved.modular.sampling.as_ref().unwrap();
    let expected_sampling = expected.modular.sampling.as_ref().unwrap();
    assert_eq!(sampling.max_channel_size, 128);
    assert_eq!(
        sampling.max_channel_size,
        expected_sampling.max_channel_size
    );
    assert_eq!(sampling.nb_repeats, expected_sampling.nb_repeats);
    assert_eq!(sampling.properties, expected_sampling.properties);
    if speed_tier == 3 && predictor == Predictor::Zero {
        assert_ne!(
            resolved.modular.source_lz77.as_ref().unwrap().method,
            options.modular.source_lz77.as_ref().unwrap().method,
            "reference LZ selection must use Gradient, not main Zero"
        );
    }
    (fixture, options)
}

#[test]
fn source_scheduler_projection_activates_add_and_resolves_atlas_geometry() {
    use rs_jxl::modular::predictor::Predictor;
    for tier in [1, 3, 5] {
        for predictor in [Predictor::Zero, Predictor::Best] {
            for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
                projected_patch_case(tier, predictor, layout);
            }
        }
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL official decoder path list"]
fn source_scheduler_add_and_deferred_reference_match_official_djxl() {
    use rs_jxl::modular::predictor::Predictor;
    let decoders = decoders();
    let scratch = ScratchDir::new();
    for decoder in &decoders {
        let banner = run(
            Command::new(decoder).arg("--version"),
            &scratch.path,
            "djxl version",
        );
        assert!(
            banner.contains("djxl v"),
            "not a djxl version banner: {banner}"
        );
        eprintln!("{}\n{banner}", decoder.display());
    }
    let (mut main_checks, mut extra_checks) = (0, 0);
    for tier in [1, 3, 5] {
        for predictor in [Predictor::Zero, Predictor::Best] {
            for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
                let (fixture, options) = projected_patch_case(tier, predictor, layout);
                let context = format!(
                    "source projected Add tier={tier} predictor={predictor:?} layout={layout:?}"
                );
                let encoded =
                    encode(&fixture.image, &options).unwrap_or_else(|e| panic!("{context}: {e}"));
                let mut disabled = options.clone();
                disabled.source_patches = None;
                disabled.source_patch_reference = None;
                assert_ne!(
                    encoded,
                    encode(&fixture.image, &disabled).unwrap(),
                    "{context}: source Add silently bypassed"
                );
                let stem = format!("projected-{tier}-{predictor:?}-{layout:?}");
                let input = scratch.path.join(format!("{stem}.jxl"));
                fs::write(&input, encoded).unwrap();
                for (decoder_index, decoder) in decoders.iter().enumerate() {
                    let context = format!("{context}, decoder={}", decoder.display());
                    let output = scratch.path.join(format!("{stem}-d{decoder_index}.png"));
                    assert!(!output.exists());
                    let mut command = Command::new(decoder);
                    command
                        .arg(&input)
                        .arg(&output)
                        .arg("--bits_per_sample=16")
                        .arg("--num_threads=1")
                        .arg("--quiet");
                    if !fixture.image.extra_planes().is_empty() {
                        command.arg("--output_extra_channels");
                    }
                    run(&mut command, &scratch.path, &context);
                    assert_png(&output, &fixture.image, &context);
                    main_checks += 1;
                    for (index, plane) in fixture.image.extra_planes().iter().enumerate() {
                        let expected = Image::new(
                            fixture.image.width(),
                            fixture.image.height(),
                            PixelLayout::Gray,
                            plane.clone(),
                        )
                        .unwrap();
                        assert_png(
                            &extra_png(&output, index),
                            &expected,
                            &format!("{context}, extra={index}"),
                        );
                        extra_checks += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "{main_checks} projected source-Add main checks and {extra_checks} extra checks passed"
    );
}
