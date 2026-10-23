// Native source scheduler integration, not mock byte-size policy tests.
// Follows reference.rs / patch_add_reference.rs oracle and cleanup patterns;
// helpers stay local so no existing test module is included or modified.
// RS_JXL_DJXL must list official djxl executables (OS path-list syntax).
// Parent runs centrally, preferably with both official 0.12 and 0.13:
// cargo test -p rs-jxl --test scheduler_frontier_reference -- --ignored --nocapture --test-threads=1
// Tests are independently selectable; preflight is the larger optional smoke.
// Raw e11 backend and the complete native tectonic tournament have separate tests.

use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::scheduler::{
    CandidateParams, DirectFrontier, DriverOptions, FrontierOptions, NativeBackendOptions,
    NativeSearchResult, PreflightOptions, SOURCE_COMMIT, ScopeRouting, WorkerLimits,
    driver::{BackendSemantics, CalibrationOptions, SearchEvent},
    native::{NativeTrialKind, encode_native_candidate},
    params::Override,
    run_native_pixel_search, run_native_tectonic,
    search::SearchLimits,
    tectonic::TectonicBranch,
};
use rs_jxl::tools::lz77::SourceLz77Method;
use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout};

const DECODER_TIMEOUT: Duration = Duration::from_secs(30);
const LARGE_WIDTH: u32 = 1024;
const LARGE_HEIGHT: u32 = 513;
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

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
                "rs-jxl-scheduler-frontier-{}-{stamp}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self { root, path },
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        }
        panic!("cannot create unique scheduler reference scratch directory");
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let valid = self.path.parent() == Some(self.root.as_path())
            && self.path.canonicalize().is_ok_and(|path| path == self.path);
        if !valid {
            eprintln!(
                "refusing cleanup of changed scratch path {}",
                self.path.display()
            );
            return;
        }
        if let Err(error) = fs::remove_dir_all(&self.path) {
            if thread::panicking() {
                eprintln!("cleanup {}: {error}", self.path.display());
            } else {
                panic!("cleanup {}: {error}", self.path.display());
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
        if start.elapsed() >= DECODER_TIMEOUT {
            child.0.kill()?;
            child.0.wait()?;
            return Ok(None);
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn run_checked(command: &mut Command, scratch: &Path, context: &str) -> String {
    for (key, _) in env::vars_os() {
        let name = key.to_string_lossy().to_ascii_uppercase();
        if name.starts_with("JXL_") || name.starts_with("RS_JXL_") {
            command.env_remove(key);
        }
    }
    let stdout = scratch.join("stdout.log");
    let stderr = scratch.join("stderr.log");
    // Files avoid pipe backpressure; close Windows handles before RAII cleanup.
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    let status = wait(command);
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let log = format!(
        "stdout:\n{}\nstderr:\n{}",
        fs::read_to_string(stdout).unwrap_or_default(),
        fs::read_to_string(stderr).unwrap_or_default()
    );
    match status {
        Ok(Some(status)) if status.success() => log,
        Ok(Some(status)) => panic!("{context}: djxl exited {status}\n{log}"),
        Ok(None) => {
            panic!("{context}: djxl exceeded {DECODER_TIMEOUT:?}, killed and reaped\n{log}")
        }
        Err(error) => panic!("{context}: launch djxl: {error}\n{log}"),
    }
}

struct Oracle {
    scratch: ScratchDir,
    decoders: Vec<PathBuf>,
}

impl Oracle {
    fn new() -> Self {
        let setting = env::var_os("RS_JXL_DJXL").expect(
            "explicit scheduler reference test requires RS_JXL_DJXL; missing oracles are not a skip",
        );
        let decoders: Vec<_> = env::split_paths(&setting)
            .map(|path| {
                assert!(
                    !path.as_os_str().is_empty() && path.is_file(),
                    "invalid official djxl path {}",
                    path.display()
                );
                path.canonicalize().expect("resolve official djxl")
            })
            .collect();
        assert!(!decoders.is_empty(), "RS_JXL_DJXL has no decoder paths");
        let scratch = ScratchDir::new();
        for decoder in &decoders {
            let context = format!("{} --version", decoder.display());
            let banner = run_checked(
                Command::new(decoder).arg("--version"),
                &scratch.path,
                &context,
            );
            assert!(
                banner.contains("djxl v"),
                "{context}: unexpected banner {banner}"
            );
            eprintln!("{context}\n{banner}");
        }
        Self { scratch, decoders }
    }

    fn check(&self, bytes: &[u8], image: &Image, context: &str) {
        assert!(!bytes.is_empty(), "{context}: empty native output");
        let input = self.scratch.path.join("retained.jxl");
        fs::write(&input, bytes).unwrap();
        for decoder in &self.decoders {
            let output = self.scratch.path.join("decoded.png");
            assert!(!output.exists(), "{context}: stale decoder output");
            let label = format!("{context}; oracle={}", decoder.display());
            run_checked(
                Command::new(decoder)
                    .arg(&input)
                    .arg(&output)
                    .arg(format!("--bits_per_sample={}", image.bits()))
                    .arg("--num_threads=1")
                    .arg("--quiet"),
                &self.scratch.path,
                &label,
            );
            assert_png_samples(&output, image, &label);
            fs::remove_file(output).unwrap();
        }
        fs::remove_file(input).unwrap();
    }
}

fn assert_png_samples(path: &Path, image: &Image, context: &str) {
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path).unwrap()));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder
        .read_info()
        .unwrap_or_else(|e| panic!("{context}: PNG header: {e}"));
    let color = match image.layout() {
        PixelLayout::Gray => png::ColorType::Grayscale,
        PixelLayout::GrayAlpha => png::ColorType::GrayscaleAlpha,
        PixelLayout::Rgb => png::ColorType::Rgb,
        PixelLayout::Rgba => png::ColorType::Rgba,
    };
    let depth = match image.bits() {
        8 => png::BitDepth::Eight,
        16 => png::BitDepth::Sixteen,
        bits => panic!("unexpected fixture bit depth {bits}"),
    };
    let info = reader.info();
    assert_eq!(
        (info.width, info.height),
        (image.width(), image.height()),
        "{context}"
    );
    assert_eq!(info.color_type, color, "{context}: channels");
    assert_eq!(info.bit_depth, depth, "{context}: precision");
    assert!(
        info.animation_control.is_none(),
        "{context}: unexpected APNG"
    );
    let expected: Vec<u16> = match image.pixels() {
        PixelData::U8(values) => values.iter().copied().map(u16::from).collect(),
        PixelData::U16(values) => values.clone(),
        _ => panic!("only integer U8/U16 fixtures belong in this oracle"),
    };
    let byte_count = expected.len() * usize::from(image.bits() / 8);
    assert_eq!(reader.output_buffer_size(), Some(byte_count), "{context}");
    let mut bytes = vec![0; byte_count];
    let output = reader
        .next_frame(&mut bytes)
        .unwrap_or_else(|e| panic!("{context}: PNG pixels: {e}"));
    assert_eq!(
        output.buffer_size(),
        byte_count,
        "{context}: decoded length"
    );
    assert_eq!(output.color_type, color, "{context}: decoded channels");
    assert_eq!(output.bit_depth, depth, "{context}: decoded precision");
    for (index, &expected) in expected.iter().enumerate() {
        // PNG uint16 is big-endian; never drop low bytes or mask alpha=0 RGB.
        let actual = if image.bits() == 16 {
            u16::from_be_bytes([bytes[index * 2], bytes[index * 2 + 1]])
        } else {
            u16::from(bytes[index])
        };
        let pixel = index / image.channel_count();
        assert_eq!(
            actual,
            expected,
            "{context}: ({}, {}) channel {}; all samples including hidden RGB",
            pixel % image.width() as usize,
            pixel / image.width() as usize,
            index % image.channel_count()
        );
    }
}

fn small_image(width: u32, height: u32, layout: PixelLayout, bits: u8) -> Image {
    let channels = match layout {
        PixelLayout::Gray => 1,
        PixelLayout::GrayAlpha => 2,
        PixelLayout::Rgb => 3,
        PixelLayout::Rgba => 4,
    };
    let max = (1u32 << bits) - 1;
    let mut values = Vec::with_capacity(width as usize * height as usize * channels);
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                let alpha = (layout == PixelLayout::Rgba || layout == PixelLayout::GrayAlpha)
                    && channel == channels - 1;
                let value = if alpha {
                    match (x + 3 * y) % 4 {
                        0 => 0,
                        1 => 1,
                        2 => max / 2,
                        _ => max,
                    }
                } else {
                    (259 + x * 31 + y * 509 + channel as u32 * 1021 + (x * y % 17) * 997) & max
                };
                values.push(value as u16);
            }
        }
    }
    if bits == 16 {
        assert!(
            values.iter().any(|value| value % 257 != 0),
            "exercise actual 16-bit low bits"
        );
    }
    if layout == PixelLayout::Rgba {
        let hidden: Vec<_> = values
            .chunks_exact(4)
            .filter(|pixel| pixel[3] == 0)
            .collect();
        assert!(hidden.len() > 1 && hidden.iter().any(|pixel| pixel[..3] != hidden[0][..3]));
        assert!(
            hidden
                .iter()
                .any(|pixel| pixel[..3].iter().any(|&v| v != 0))
        );
    }
    let pixels = if bits == 8 {
        PixelData::U8(
            values
                .into_iter()
                .map(|value| u8::try_from(value).unwrap())
                .collect(),
        )
    } else {
        PixelData::U16(values)
    };
    Image::new(width, height, layout, pixels).unwrap()
}

fn sparse_gray16() -> Image {
    // Just above 2^19, below 2^21: activates compact/strip/scan gates without
    // a megapixel RGB workload. Sparse, distinct markers avoid a constant image.
    let mut values = vec![259u16; LARGE_WIDTH as usize * LARGE_HEIGHT as usize];
    for y in (0..LARGE_HEIGHT).step_by(19) {
        for x in (0..LARGE_WIDTH).step_by(31) {
            values[(y * LARGE_WIDTH + x) as usize] = (1 + (x * 977 + y * 7919) % 65534) as u16;
        }
    }
    values[0] = 0;
    values[1] = 1;
    values[2] = 65534;
    *values.last_mut().unwrap() = 65535;
    let image = Image::new(
        LARGE_WIDTH,
        LARGE_HEIGHT,
        PixelLayout::Gray,
        PixelData::U16(values),
    )
    .unwrap();
    let pixels = u64::from(image.width()) * u64::from(image.height());
    assert!(pixels > 1 << 19 && pixels < 2 << 20);
    image
}

fn route_options(frontier: FrontierOptions, max_full_encodes: usize) -> DriverOptions {
    DriverOptions {
        frontier,
        // Bound work, not candidate sampling or tree policy; limits must fail,
        // never turn a partial search into a successful reference test.
        full_limits: SearchLimits {
            max_full_encodes,
            max_full_pixels: max_full_encodes as u64
                * u64::from(LARGE_WIDTH)
                * u64::from(LARGE_HEIGHT),
            max_output_bytes: 8 << 20,
        },
        workers: WorkerLimits {
            tasks: 1,
            scratch_bytes: 256 << 20,
        },
        ..Default::default()
    }
}

fn native_search(image: &Image, options: &DriverOptions, context: &str) -> NativeSearchResult {
    native_search_with_threads(image, options, 1, context)
}

fn native_search_with_threads(
    image: &Image,
    options: &DriverOptions,
    threads: usize,
    context: &str,
) -> NativeSearchResult {
    let params = CandidateParams::default().backend(9).unwrap();
    let result = run_native_pixel_search(
        image,
        &params,
        options,
        &EncodeOptions {
            threads,
            max_output_bytes: 8 << 20,
            ..Default::default()
        },
        NativeBackendOptions::default(),
    )
    .unwrap_or_else(|error| panic!("{context}: strict native search failed: {error}"));
    let output = &result.output;
    let report = &output.report;
    assert_eq!(report.source_commit, SOURCE_COMMIT, "{context}");
    assert_eq!(
        report.backend_semantics,
        BackendSemantics::Unverified,
        "{context}: no equivalence certification"
    );
    assert!(
        !result.adaptations.is_empty(),
        "{context}: no real native calls"
    );
    for adaptation in &result.adaptations {
        assert!(
            adaptation.gaps.is_empty(),
            "{context}: unsupported native projection: {:?}",
            adaptation.gaps
        );
        assert_eq!(
            adaptation.requested, adaptation.effective,
            "{context}: policy rewritten"
        );
    }
    assert!(
        output.artifact.gaps.is_empty(),
        "{context}: winner has projection gaps"
    );
    assert_eq!(
        output.params, output.artifact.effective_params,
        "{context}: artifact is not the winner"
    );
    let winner = &report.trials[report.winner_index];
    assert_eq!(winner.params, output.params, "{context}: retained params");
    assert_eq!(
        winner.bytes,
        Some(output.bytes.len()),
        "{context}: retained size"
    );
    assert_eq!(
        winner.lower_bound, None,
        "{context}: a bound is not an encoded winner"
    );
    let full = result
        .adaptations
        .iter()
        .filter(|a| a.kind == NativeTrialKind::Full)
        .count();
    let probes = result
        .adaptations
        .iter()
        .filter(|a| a.kind == NativeTrialKind::Probe)
        .count();
    assert_eq!(
        report.full_encodes, full,
        "{context}: real full encode count"
    );
    assert_eq!(report.probe_encodes, probes, "{context}: real probe count");
    assert_eq!(
        report.full_pixels,
        full as u64 * u64::from(image.width()) * u64::from(image.height()),
        "{context}"
    );
    assert_eq!(
        report.probe_pixels,
        result
            .adaptations
            .iter()
            .filter(|a| a.kind == NativeTrialKind::Probe)
            .map(|a| u64::from(a.width) * u64::from(a.height))
            .sum::<u64>(),
        "{context}"
    );
    assert_eq!(report.trials.len(), full, "{context}: trial transcript");
    let routes: Vec<_> = report
        .events
        .iter()
        .filter_map(|event| match event {
            SearchEvent::Route { name, candidates } => Some((*name, *candidates)),
            _ => None,
        })
        .collect();
    eprintln!(
        "{context}: full={full} probes={probes} winner={} bytes={} routes={routes:?}",
        report.winner_index,
        output.bytes.len()
    );
    result
}

fn require_route(result: &NativeSearchResult, name: &str, context: &str) -> usize {
    let routes: Vec<_> = result
        .output
        .report
        .events
        .iter()
        .filter_map(|event| match event {
            SearchEvent::Route {
                name: actual,
                candidates,
            } if *actual == name => Some(*candidates),
            _ => None,
        })
        .collect();
    assert_eq!(
        routes.len(),
        1,
        "{context}: requested route did not run exactly once: {name}"
    );
    assert!(routes[0] > 0, "{context}: empty route {name}");
    routes[0]
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; actual global tournament on >4096 pixels"]
fn global_tournament_hidden_rgba_rgb_gray16_match_official_djxl() {
    let oracle = Oracle::new();
    let options = route_options(
        FrontierOptions {
            global_tournament: true,
            ..Default::default()
        },
        9,
    );
    for (layout, bits) in [
        (PixelLayout::Rgba, 8),
        (PixelLayout::Rgba, 16),
        (PixelLayout::Rgb, 16),
        (PixelLayout::Gray, 16),
    ] {
        let image = small_image(67, 67, layout, bits);
        assert!(image.width() * image.height() > 4096);
        let context = format!("global tournament {layout:?}/{bits}");
        let result = native_search(&image, &options, &context);
        assert_eq!(require_route(&result, "global-tournament", &context), 8);
        let report = &result.output.report;
        assert_eq!(
            report.full_encodes, 9,
            "{context}: eight e7 witnesses and one e9 finalist"
        );
        assert_eq!(
            report.probe_encodes, 0,
            "{context}: global tournament does not crop"
        );
        assert!(
            report.trials[..8]
                .iter()
                .all(|t| t.calibration && t.params.speed_tier == 3)
        );
        assert!(!report.trials[8].calibration);
        assert_eq!(report.trials[8].params.speed_tier, 1);
        // A smaller calibration witness may legitimately remain the winner.
        oracle.check(&result.output.bytes, &image, &context);
    }
}

fn compact_roundtrip(direct: DirectFrontier) {
    let oracle = Oracle::new();
    let image = sparse_gray16();
    let options = route_options(
        FrontierOptions {
            direct,
            compact: true,
            ..Default::default()
        },
        2,
    );
    let context = format!(
        "compact {direct:?} sparse Gray16 {}x{}",
        image.width(),
        image.height()
    );
    let result = native_search(&image, &options, &context);
    let finalists = require_route(&result, "direct-frontier", &context);
    assert!((1..=2).contains(&finalists));
    assert!((1..=finalists).contains(&result.output.report.full_encodes));
    assert!(result.output.report.probe_encodes > 0 && result.output.report.probe_pixels > 0);
    // The compact branch intentionally skips full-frame calibration, even
    // when the noncompact direct enum would request calibrated/shrink/dense.
    assert!(result.output.report.trials.iter().all(|t| !t.calibration));
    oracle.check(&result.output.bytes, &image, &context);
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; >512k sparse compact route"]
fn direct_sampled_compact_gray16_matches_official_djxl() {
    compact_roundtrip(DirectFrontier::Sampled);
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; >512k sparse compact route"]
fn direct_calibrated_compact_gray16_matches_official_djxl() {
    compact_roundtrip(DirectFrontier::Calibrated);
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; >512k sparse compact route"]
fn direct_extended_compact_gray16_matches_official_djxl() {
    compact_roundtrip(DirectFrontier::Extended);
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; >512k sparse compact route"]
fn direct_shrink_compact_gray16_matches_official_djxl() {
    compact_roundtrip(DirectFrontier::Shrink);
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; >512k sparse compact route"]
fn direct_dense_compact_gray16_matches_official_djxl() {
    compact_roundtrip(DirectFrontier::Dense);
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; bounded native cooperative compact route"]
fn direct_compact_worker_options_gray16_matches_official_djxl() {
    let oracle = Oracle::new();
    let image = sparse_gray16();
    let options = DriverOptions {
        cooperative_probes: true,
        data_parallel_probes: true,
        longest_first_probes: true,
        probe_workers: Some(2),
        workers: WorkerLimits {
            tasks: 3,
            scratch_bytes: 192 << 20,
        },
        ..route_options(
            FrontierOptions {
                direct: DirectFrontier::Sampled,
                compact: true,
                parallel_finalists: true,
                ..Default::default()
            },
            2,
        )
    };
    let context = "compact Gray16 cooperative probes and parallel finalists, two threads";
    let result = native_search_with_threads(&image, &options, 2, context);
    let finalists = require_route(&result, "direct-frontier", context);
    let report = &result.output.report;
    assert!((1..=2).contains(&finalists));
    assert!((1..=finalists).contains(&report.full_encodes));
    assert!(report.probe_encodes > 0 && report.probe_pixels > 0);
    assert!(
        report.screen_batches > 0,
        "{context}: no native probe batches"
    );
    assert!(report.trials.iter().all(|trial| !trial.calibration));
    assert!(result.adaptations.iter().any(|adaptation| {
        adaptation.kind == NativeTrialKind::Probe
            && u64::from(adaptation.width) * u64::from(adaptation.height)
                < u64::from(image.width()) * u64::from(image.height())
    }));
    // Compact ranking uses Rounds: cooperative dispatch and longest-first apply.
    // Cooperative takes priority over data-parallel; parallel finalists need
    // two distinct nominations, so neither conditional path is claimed here.
    eprintln!(
        "{context}: finalists={finalists} batches={}",
        report.screen_batches
    );
    oracle.check(&result.output.bytes, &image, context);
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; >512k native full-width strip probes"]
fn strip_tournament_sparse_gray16_matches_official_djxl() {
    let oracle = Oracle::new();
    let image = sparse_gray16();
    let options = route_options(
        FrontierOptions {
            strip_tournament: true,
            ..Default::default()
        },
        2,
    );
    let context = "strip tournament sparse Gray16";
    let result = native_search(&image, &options, context);
    assert!((1..=2).contains(&require_route(&result, "strip-tournament", context)));
    assert!((1..=2).contains(&result.output.report.full_encodes));
    assert!(
        result
            .adaptations
            .iter()
            .any(|a| a.kind == NativeTrialKind::Probe
                && a.width == image.width()
                && a.height < image.height()),
        "full-width strip was not encoded"
    );
    assert!(result.output.report.probe_pixels > 0);
    oracle.check(&result.output.bytes, &image, context);
}

#[test]
#[ignore = "optional larger preflight/calibration/Broad configuration; parent budgets runtime"]
fn preflight_calibration_broad_sparse_gray16_matches_official_djxl() {
    let oracle = Oracle::new();
    let image = sparse_gray16();
    let options = DriverOptions {
        coverage_scan: true,
        calibration: CalibrationOptions {
            enabled: true,
            fast: true,
            routes: true,
            refinements: true,
        },
        preflight: PreflightOptions {
            enabled: true,
            scope: ScopeRouting::Broad,
            calibration_reuse: true,
            ..Default::default()
        },
        ..route_options(FrontierOptions::default(), 32)
    };
    let context = "preflight enabled + fast calibration + Broad sparse Gray16";
    let result = native_search(&image, &options, context);
    assert!(result.output.report.full_encodes > 1);
    assert!(result.output.report.probe_encodes > 0 && result.output.report.probe_pixels > 0);
    assert!(
        result
            .output
            .report
            .events
            .iter()
            .any(|e| matches!(e, SearchEvent::Screened { .. })),
        "preflight candidate screening did not execute"
    );
    // Calibration is data/score-gated. This is an enabled-configuration smoke,
    // not a claim that every calibration or broad-scope sub-branch activated.
    let calibrations = result
        .output
        .report
        .trials
        .iter()
        .filter(|t| t.calibration)
        .count();
    eprintln!("{context}: observed {calibrations} full calibration witnesses");
    oracle.check(&result.output.bytes, &image, context);
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; complete native tectonic on tiny hidden RGBA16"]
fn native_tectonic_tiny_hidden_rgba16_matches_official_djxl() {
    let oracle = Oracle::new();
    let image = small_image(8, 8, PixelLayout::Rgba, 16);
    let params = CandidateParams::default().backend(11).unwrap();
    let context = "native tectonic complete tournament, 8x8 hidden RGBA16";
    let result = run_native_tectonic(
        &image,
        &params,
        SearchLimits {
            max_full_encodes: 27,
            max_full_pixels: 27 * 64,
            max_output_bytes: 1 << 20,
        },
        WorkerLimits {
            tasks: 2,
            scratch_bytes: 96 << 20,
        },
        &EncodeOptions {
            threads: 2,
            max_output_bytes: 1 << 20,
            ..Default::default()
        },
        NativeBackendOptions::default(),
    )
    .unwrap_or_else(|error| panic!("{context}: strict native tournament failed: {error}"));
    let output = &result.output;
    let report = &output.report;
    let expected_encodes = match report.branch {
        TectonicBranch::LessPalette => 27,
        TectonicBranch::MorePalette => 23,
    };
    assert_eq!(report.full_encodes, expected_encodes, "{context}");
    assert_eq!(
        report.full_pixels,
        expected_encodes as u64 * 64,
        "{context}"
    );
    assert_eq!(
        report.trials.len() + 1,
        expected_encodes,
        "{context}: final re-encode"
    );
    assert_eq!(
        result.adaptations.len(),
        expected_encodes,
        "{context}: actual native calls"
    );
    for adaptation in &result.adaptations {
        assert_eq!(adaptation.kind, NativeTrialKind::Full, "{context}");
        assert_eq!((adaptation.width, adaptation.height), (8, 8), "{context}");
        assert!(adaptation.gaps.is_empty(), "{context}: projection gaps");
        assert_eq!(
            adaptation.requested, adaptation.effective,
            "{context}: policy rewrite"
        );
        // Source tectonic dispatches e10 candidates, unlike the separate raw11 test.
        assert_eq!(adaptation.effective.speed_tier, 0, "{context}");
    }
    assert!(report.trials.iter().all(|trial| trial.bytes > 0));
    assert_eq!(
        output.params, report.trials[report.winner_index].params,
        "{context}"
    );
    assert_eq!(
        output.params,
        result.adaptations.last().unwrap().requested,
        "{context}: final call"
    );
    assert_eq!(
        output.params, output.artifact.effective_params,
        "{context}: final artifact"
    );
    assert!(output.artifact.gaps.is_empty(), "{context}");
    assert_eq!(
        report.final_bytes,
        output.bytes.len(),
        "{context}: final size accounting"
    );
    eprintln!(
        "{context}: branch={:?} full={} winner={}",
        report.branch, report.full_encodes, report.winner_index
    );
    // No byte identity requirement across trials, worker modes, or raw11.
    oracle.check(&output.bytes, &image, context);
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; e11 RAW backend only, not the e11 tournament"]
fn effort11_raw_backend_small_integer_pixels_match_official_djxl() {
    let oracle = Oracle::new();
    let params = CandidateParams {
        palette_colors: 0,
        channel_colors_percent: 0.0,
        channel_colors_pre_transform_percent: 0.0,
        patches: Override::Off,
        ..CandidateParams::default().backend(11).unwrap()
    };
    assert_eq!(params.speed_tier, -1);
    for bits in [8, 16] {
        for layout in [PixelLayout::Gray, PixelLayout::Rgb, PixelLayout::Rgba] {
            let image = small_image(23, 17, layout, bits);
            let context = format!("e11 single raw backend {layout:?}/{bits}, no tournament");
            let result = encode_native_candidate(
                &params,
                &image,
                &EncodeOptions {
                    threads: 1,
                    max_output_bytes: 8 << 20,
                    ..Default::default()
                },
                NativeBackendOptions::default(),
            )
            .unwrap_or_else(|error| panic!("{context}: strict native backend: {error}"));
            assert!(
                result.artifact.gaps.is_empty(),
                "{context}: projection gaps"
            );
            assert_eq!(
                result.artifact.effective_params, params,
                "{context}: tier/policy rewrite"
            );
            let modular = &result.artifact.options.modular;
            let palette = modular
                .palette_policy
                .as_ref()
                .expect("source palette policy");
            assert_eq!(
                palette.speed_tier, -1,
                "{context}: must not silently use e10"
            );
            assert_eq!(palette.palette_colors, 0);
            assert_eq!(palette.channel_colors_percent, 0.0);
            assert_eq!(palette.channel_colors_pre_transform_percent, 0.0);
            assert_eq!(
                modular.source_lz77.as_ref().unwrap().method,
                SourceLz77Method::Optc256
            );
            assert_eq!(modular.sampling.as_ref().unwrap().node_threshold, 61.0);
            assert!(result.artifact.options.source_patches.is_none());
            oracle.check(&result.bytes, &image, &context);
        }
    }
}
