//! Independent stage-one parallel execution acceptance, not a JXL byte oracle.
//! Parent runs centrally in release mode with RS_JXL_DJXL containing official
//! 0.12 and 0.13 decoder paths (an OS path list, ';' on Windows).
//!
//! Each encode re-enters this test binary with RS_JXL_PROFILE=1. This captures
//! real dispatch without unsafe environment changes or private library hooks.
//! The child runs the Rust encoder; only decoding uses an external codec.
//! No changes to reference.rs or shared test support are required.

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::entropy::{hybrid_uint::HybridUintConfig, options::EntropyOptions};
use rs_jxl::modular::{
    encode::FixedTree,
    ma_train::{ChannelTreeMode, MAX_LEAVES, ParallelOptions},
    predictor::Predictor,
    sample::{SamplingPolicy, SourcePolicy},
};
use rs_jxl::scheduler::{
    CandidateParams, DirectFrontier, DriverOptions, FrontierOptions, NativeBackendOptions,
    SOURCE_COMMIT, WorkerLimits, driver::SearchEvent, native::NativeTrialKind,
    run_native_pixel_search, search::SearchLimits,
};
use rs_jxl::tools::lz77::{
    SourceBlockMode, SourceBucketSelection, SourceConfig, SourceIndexMode, SourceLz77Method,
    SourceMatchCost, SourceParallelism,
};
// The fixture module's own tests use these crate-relative public module names.
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode, icc, jpeg};

#[allow(dead_code)]
#[path = "support/jpeg_fixtures.rs"]
mod jpeg_fixtures;

const CHILD_CASE: &str = "RS_JXL_PARALLEL_TEST_CHILD";
const CHILD_OUTPUT: &str = "RS_JXL_PARALLEL_TEST_OUTPUT";
const OUTPUT_LIMIT: usize = 8 << 20;
const LOG_LIMIT: u64 = 16 << 20;
const LZ_WIDTH: u32 = 384;
const LZ_HEIGHT: u32 = 256;
const SEARCH_WIDTH: u32 = 1024;
const SEARCH_HEIGHT: u32 = 513;
const MA_WIDTH: u32 = 192;
const MA_HEIGHT: u32 = 256;
const JPEG_CASES: [&str; 2] = ["ycbcr444_seq_513x257_rst7", "ycbcr420_seq_600x400_rst7"];
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug)]
struct Case {
    threads: usize,
    quiet: bool,
}

const SERIAL: Case = Case {
    threads: 1,
    quiet: false,
};
const PARALLEL: Case = Case {
    threads: 4,
    quiet: false,
};
const QUIET: Case = Case {
    threads: 4,
    quiet: true,
};

fn ma_image() -> Image {
    // The channel_tree_reference.rs fixture's unique channel/x/y records:
    // 49,152 per channel, 24,576 per row half, 12,288 per second row half.
    let mut state = 0x79f5_6d2bu32;
    let mut samples = Vec::with_capacity(MA_WIDTH as usize * MA_HEIGHT as usize * 4);
    for y in 0..MA_HEIGHT {
        for x in 0..MA_WIDTH {
            for channel in 0..4 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                samples.push(match channel {
                    1 if y < 64 => (x + y * 3) as u8,
                    2 if (128..192).contains(&y) => (x * 5 + y) as u8,
                    3 if (y * MA_WIDTH + x).is_multiple_of(17) => 0,
                    _ => state as u8,
                });
            }
        }
    }
    samples[..4].copy_from_slice(&[17, 223, 91, 0]);
    Image::new(
        MA_WIDTH,
        MA_HEIGHT,
        PixelLayout::Rgba,
        PixelData::U8(samples),
    )
    .unwrap()
}

fn ma_options(case: Case) -> EncodeOptions {
    let sampling = SamplingPolicy::from_source(&SourcePolicy {
        speed_tier: 1,
        predictor: Predictor::Gradient,
        learning_fraction: 1.0,
        num_extra_channels: 1,
        ..Default::default()
    })
    .unwrap();
    EncodeOptions {
        threads: case.threads,
        shared_tree: true,
        max_pixels: u64::from(MA_WIDTH) * u64::from(MA_HEIGHT),
        max_output_bytes: OUTPUT_LIMIT,
        modular: ModularOptions {
            predictor: Predictor::Gradient,
            rct: None,
            lz77: false,
            sampling: Some(sampling),
            sample_limit: MA_WIDTH as usize * MA_HEIGHT as usize * 4,
            max_tree_leaves: MAX_LEAVES,
            source_channel_tree: ChannelTreeMode::ChannelsAndRowsTwice,
            source_ma_parallel: ParallelOptions {
                workers: WorkerLimits {
                    tasks: 4,
                    scratch_bytes: 64 << 20,
                },
                quiet: case.quiet,
                ..Default::default()
            },
            source_tree_entropy: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn subtree_options(case: Case) -> EncodeOptions {
    let mut options = ma_options(case);
    // One 196,608-record learner exceeds the ordinary 65,536-record gate.
    // Disable channel roots so the arena note can only come from detached
    // whole subtrees, not from the already-covered channel/row wrapper.
    options.modular.source_channel_tree = ChannelTreeMode::Disabled;
    options.modular.source_ma_parallel.subtrees = true;
    options.modular.source_ma_parallel.balanced = false;
    options.modular.source_ma_parallel.small_subtrees = false;
    options
}

fn lz_image() -> Image {
    let mut samples = Vec::with_capacity(LZ_WIDTH as usize * LZ_HEIGHT as usize * 4);
    for y in 0..LZ_HEIGHT {
        for x in 0..LZ_WIDTH {
            let (tx, ty) = (x % 32, y % 32);
            samples.extend_from_slice(&[
                (tx * 17 + ty * 29) as u8,
                (tx * tx + ty * 7) as u8,
                (tx * 3 + ty * ty * 5) as u8,
                if (tx + ty).is_multiple_of(7) {
                    0
                } else {
                    (31 + tx * 11 + ty * 13) as u8
                },
            ]);
        }
    }
    samples[..4].copy_from_slice(&[17, 223, 91, 0]);
    Image::new(
        LZ_WIDTH,
        LZ_HEIGHT,
        PixelLayout::Rgba,
        PixelData::U8(samples),
    )
    .unwrap()
}

fn lz_options(case: Case) -> EncodeOptions {
    let length = HybridUintConfig::new(0, 0, 0).unwrap();
    EncodeOptions {
        threads: case.threads,
        shared_tree: true,
        group_size_shift: 1,
        max_pixels: u64::from(LZ_WIDTH) * u64::from(LZ_HEIGHT),
        max_output_bytes: OUTPUT_LIMIT,
        modular: ModularOptions {
            predictor: Predictor::Gradient,
            rct: None,
            fixed_tree: Some(FixedTree::GradientLeaf),
            source_lz77: Some(SourceConfig {
                method: SourceLz77Method::Optc256,
                bucket_selection: Some(SourceBucketSelection::Parallel32),
                match_cost: SourceMatchCost::Frontier,
                index_mode: SourceIndexMode::All,
                parallel_index_build: true,
                block_mode: SourceBlockMode::Intervals,
                length_uint_config: length,
                parallelism: Some(SourceParallelism {
                    max_workers: 4,
                    scratch_bytes: 64 << 20,
                    fast_streams: false,
                }),
                stream_arena: true,
                stream_arena_quiet: case.quiet,
                // Keep the source savings/index/seed gates, not diagnostic
                // require_global_savings=false or a shared-seed-index bypass.
                ..Default::default()
            }),
            entropy_options: EntropyOptions {
                lz77_length_config: length,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn scheduler_image() -> Image {
    // Strictly above the source compact frontier's 2^19 pixel gate.
    let mut samples = vec![259u16; SEARCH_WIDTH as usize * SEARCH_HEIGHT as usize];
    for y in (0..SEARCH_HEIGHT).step_by(19) {
        for x in (0..SEARCH_WIDTH).step_by(31) {
            samples[(y * SEARCH_WIDTH + x) as usize] = (1 + (x * 977 + y * 7919) % 65534) as u16;
        }
    }
    samples[..3].copy_from_slice(&[0, 1, 65534]);
    *samples.last_mut().unwrap() = 65535;
    Image::new(
        SEARCH_WIDTH,
        SEARCH_HEIGHT,
        PixelLayout::Gray,
        PixelData::U16(samples),
    )
    .unwrap()
}

fn encode_scheduler(image: &Image, case: Case) -> Vec<u8> {
    let pixels = u64::from(image.width()) * u64::from(image.height());
    let options = DriverOptions {
        cooperative_probes: true,
        arena_quiet: case.quiet,
        data_parallel_probes: true,
        longest_first_probes: true,
        probe_workers: Some(2),
        workers: WorkerLimits {
            tasks: 3,
            scratch_bytes: 192 << 20,
        },
        frontier: FrontierOptions {
            direct: DirectFrontier::Sampled,
            compact: true,
            // Compact Rounds provide a real cooperative batch. Two distinct
            // full finalists are not guaranteed, so do not claim that route.
            parallel_finalists: false,
            ..Default::default()
        },
        full_limits: SearchLimits {
            max_full_encodes: 2,
            max_full_pixels: 2 * pixels,
            max_output_bytes: OUTPUT_LIMIT,
        },
        ..Default::default()
    };
    let result = run_native_pixel_search(
        image,
        &CandidateParams::default().backend(9).unwrap(),
        &options,
        &EncodeOptions {
            threads: case.threads,
            max_output_bytes: OUTPUT_LIMIT,
            ..Default::default()
        },
        NativeBackendOptions::default(),
    )
    .unwrap();
    let output = result.output;
    let report = &output.report;
    assert_eq!(report.source_commit, SOURCE_COMMIT);
    assert!(report.events.iter().any(|event| matches!(
        event,
        SearchEvent::Route {
            name: "direct-frontier",
            candidates: 1..=2
        }
    )));
    assert!((1..=2).contains(&report.full_encodes));
    assert!(report.probe_encodes > 0 && report.probe_pixels > 0 && report.screen_batches > 0);
    assert!(report.trials.iter().all(|trial| !trial.calibration));
    assert!(
        result
            .adaptations
            .iter()
            .any(|a| a.kind == NativeTrialKind::Probe
                && u64::from(a.width) * u64::from(a.height) < pixels)
    );
    for adaptation in &result.adaptations {
        assert!(
            adaptation.gaps.is_empty(),
            "native policy projection: {:?}",
            adaptation.gaps
        );
        assert_eq!(adaptation.requested, adaptation.effective);
    }
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
    assert_eq!(report.full_encodes, full);
    assert_eq!(report.full_pixels, full as u64 * pixels);
    assert_eq!(report.probe_encodes, probes);
    assert_eq!(
        report.probe_pixels,
        result
            .adaptations
            .iter()
            .filter(|a| a.kind == NativeTrialKind::Probe)
            .map(|a| u64::from(a.width) * u64::from(a.height))
            .sum::<u64>()
    );
    assert!(output.artifact.gaps.is_empty());
    assert_eq!(output.params, output.artifact.effective_params);
    let winner = &report.trials[report.winner_index];
    assert_eq!(winner.params, output.params);
    assert_eq!(winner.bytes, Some(output.bytes.len()));
    assert_eq!(winner.lower_bound, None);
    eprintln!(
        "[parallel acceptance] scheduler: full={full} probes={probes} batches={}",
        report.screen_batches
    );
    output.bytes
}

fn jpeg_options(case: Case) -> jpeg::TranscodeOptions {
    jpeg::TranscodeOptions {
        effort: 9,
        threads: case.threads,
        cooperative_ma: Some(WorkerLimits {
            tasks: 4,
            scratch_bytes: 64 << 20,
        }),
        limits: jpeg::TranscodeLimits {
            max_output_bytes: OUTPUT_LIMIT,
            max_coefficients: 2 << 20,
            max_tokens: 2 << 20,
            sample_limit: 1 << 20,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn profile_rows<'a>(log: &'a str, stage: &str) -> Vec<BTreeMap<&'a str, usize>> {
    let prefix = format!("[rs-jxl profile] {stage}: ");
    log.lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .map(|fields| {
            fields
                .split_whitespace()
                .map(|field| {
                    let (key, value) = field.split_once('=').expect("profile key=value");
                    let value = match value {
                        "false" => 0,
                        "true" => 1,
                        _ => value.parse().expect("integer or boolean profile field"),
                    };
                    (key, value)
                })
                .collect()
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArenaRoute {
    MaChannels,
    MaSubtrees,
    LzStreams,
    JpegMa,
    Scheduler,
}

fn assert_cooperative_dispatch(log: &str, case: Case, route: ArenaRoute) {
    let rows = profile_rows(log, "execution.cooperative");
    let context = format!("{route:?} {case:?}");
    // Pixel/JPEG threads=1 means a null runner, while the scheduler creates
    // an installed one-thread pool and may legitimately enter the arena.
    if case.threads == 1 && route != ArenaRoute::Scheduler {
        assert!(
            rows.is_empty(),
            "{context}: null runner entered an arena\n{log}"
        );
        return;
    }
    assert!(
        !rows.is_empty(),
        "{context}: cooperative dispatch missing\n{log}"
    );
    let worker_slots = if route == ArenaRoute::Scheduler { 3 } else { 4 };
    let mut top_level = 0;
    let mut multi_job = false;
    for row in rows {
        let jobs = row["jobs"];
        let roots = row["root_limit"];
        assert!(
            jobs > 0 && (1..=jobs).contains(&roots),
            "{context}: invalid job/root counts\n{log}"
        );
        assert_eq!(
            row["workers"],
            case.threads.min(worker_slots),
            "{context}: worker width\n{log}"
        );
        assert_eq!(
            row["quiet"],
            usize::from(case.quiet),
            "{context}: quiet mode\n{log}"
        );
        assert!(row["nested"] <= 1, "{context}: invalid nesting flag\n{log}");
        if row["nested"] == 1 {
            // Nested groups retain the outer executor's width and quiet
            // policy, but have their own job count and local root budget.
            continue;
        }
        top_level += 1;
        multi_job |= jobs > 1;
        match route {
            ArenaRoute::MaChannels => {
                assert_eq!(jobs, 16, "{context}: channel/row jobs\n{log}");
                assert_eq!(roots, 2, "{context}: MA scratch cap\n{log}");
            }
            ArenaRoute::MaSubtrees => {
                assert!(
                    jobs > 1,
                    "{context}: multiple whole-subtree jobs required\n{log}"
                );
                assert_eq!(roots, 2, "{context}: subtree scratch cap\n{log}");
            }
            ArenaRoute::LzStreams => {
                assert_eq!(jobs, 2, "{context}: nonempty shared streams\n{log}");
                assert_eq!(roots, 2, "{context}: stream scratch cap\n{log}");
            }
            ArenaRoute::JpegMa => {
                assert!(
                    jobs > 1,
                    "{context}: multiple JPEG learner roots required\n{log}"
                );
                assert_eq!(roots, 2, "{context}: JPEG learner scratch cap\n{log}");
            }
            ArenaRoute::Scheduler => {
                assert_eq!(roots, jobs.min(3), "{context}: probe root cap\n{log}");
            }
        }
    }
    assert!(
        top_level > 0,
        "{context}: nested groups alone do not prove arena entry\n{log}"
    );
    assert!(
        multi_job,
        "{context}: no actual multi-job cooperative batch\n{log}"
    );
}

struct FailureLog<'a>(&'a str);

impl Drop for FailureLog<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            eprintln!("encoder diagnostics on failure:\n{}", self.0);
        }
    }
}

fn report_case(context: &str, input: &Path, log: &str, elapsed: Duration) {
    let rows = profile_rows(log, "execution.cooperative");
    let top = rows.iter().filter(|row| row["nested"] == 0).count();
    let max_workers = rows.iter().map(|row| row["workers"]).max().unwrap_or(0);
    let max_roots = rows.iter().map(|row| row["root_limit"]).max().unwrap_or(0);
    let subtree_jobs: usize = profile_rows(log, "ma.subtrees")
        .iter()
        .filter_map(|row| row.get("jobs"))
        .sum();
    eprintln!(
        "{context}: bytes={} encode={:.3}s arena_top={top} nested={} workers={max_workers} root_limit={max_roots} subtree_jobs={subtree_jobs}",
        fs::metadata(input).unwrap().len(),
        elapsed.as_secs_f64(),
        rows.len() - top,
    );
}

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
                "rs-jxl-parallel-{}-{stamp}-{id}",
                std::process::id()
            ));
            assert_eq!(path.parent(), Some(root.as_path()));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        }
        panic!("cannot create unique parallel oracle directory");
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

fn bounded_log(path: &Path) -> String {
    let mut bytes = Vec::new();
    File::open(path)
        .unwrap()
        .take(LOG_LIMIT + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes.len() as u64 <= LOG_LIMIT, "oracle log limit");
    String::from_utf8_lossy(&bytes).into_owned()
}

fn run_checked(command: &mut Command, scratch: &Path, context: &str, timeout: Duration) -> String {
    let stdout = scratch.join("stdout.log");
    let stderr = scratch.join("stderr.log");
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    let mut child = ReapOnDrop(
        command
            .spawn()
            .unwrap_or_else(|e| panic!("{context}: spawn: {e}")),
    );
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.0.try_wait().expect("poll child") {
            break status;
        }
        assert!(
            start.elapsed() < timeout,
            "{context}: timeout {timeout:?}; child killed/reaped on unwind"
        );
        for path in [&stdout, &stderr] {
            assert!(
                fs::metadata(path).unwrap().len() <= LOG_LIMIT,
                "{context}: child log limit"
            );
        }
        thread::sleep(Duration::from_millis(10));
    };
    let log = format!(
        "stdout:\n{}\nstderr:\n{}",
        bounded_log(&stdout),
        bounded_log(&stderr)
    );
    assert!(status.success(), "{context}: child exited {status}\n{log}");
    log
}

struct Oracle {
    scratch: ScratchDir,
    decoders: Vec<PathBuf>,
}

impl Oracle {
    fn new() -> Self {
        let scratch = ScratchDir::new();
        let value = env::var_os("RS_JXL_DJXL").expect(
            "explicit oracle requires RS_JXL_DJXL official 0.12 and 0.13 decoder path list",
        );
        let decoders: Vec<_> = env::split_paths(&value)
            .map(|path| {
                assert!(
                    !path.as_os_str().is_empty() && path.is_file(),
                    "invalid decoder {}",
                    path.display()
                );
                path.canonicalize().unwrap()
            })
            .collect();
        let mut versions = [false; 2];
        for decoder in &decoders {
            let banner = run_checked(
                Command::new(decoder).arg("--version"),
                &scratch.0,
                "decoder version",
                Duration::from_secs(30),
            );
            assert!(
                banner.contains("djxl v"),
                "not an official djxl banner: {banner}"
            );
            versions[0] |= banner.contains("djxl v0.12.");
            versions[1] |= banner.contains("djxl v0.13.");
            let version = banner.lines().find(|line| line.contains("djxl v")).unwrap();
            eprintln!("{}: {version}", decoder.display());
        }
        assert_eq!(
            versions,
            [true, true],
            "acceptance requires both official 0.12 and 0.13"
        );
        Self { scratch, decoders }
    }

    fn encode(&self, test: &str, index: usize, context: &str) -> (PathBuf, String, Duration) {
        let input = self.scratch.0.join(format!("case-{index}.jxl"));
        let started = Instant::now();
        let log = run_checked(
            Command::new(env::current_exe().unwrap())
                .args([
                    "--exact",
                    test,
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_CASE, index.to_string())
                .env(CHILD_OUTPUT, &input)
                .env("RS_JXL_PROFILE", "1"),
            &self.scratch.0,
            context,
            Duration::from_secs(240),
        );
        let size = fs::metadata(&input).expect("child output").len();
        assert!(
            size > 0 && size <= OUTPUT_LIMIT as u64,
            "{context}: output budget"
        );
        (input, log, started.elapsed())
    }

    fn pixels(&self, input: &Path, expected: &Image, context: &str) {
        for (index, decoder) in self.decoders.iter().enumerate() {
            let output = self.scratch.0.join(format!("decoded-{index}.png"));
            assert!(!output.exists(), "stale decoder output");
            let label = format!("{context}; decoder={}", decoder.display());
            run_checked(
                Command::new(decoder)
                    .arg(input)
                    .arg(&output)
                    .arg(format!("--bits_per_sample={}", expected.bits()))
                    .arg("--num_threads=1")
                    .arg("--quiet"),
                &self.scratch.0,
                &label,
                Duration::from_secs(30),
            );
            assert_png_samples(&output, expected, &label);
            fs::remove_file(output).unwrap();
        }
    }

    fn original_jpeg(&self, input: &Path, expected: &[u8], context: &str) {
        for (index, decoder) in self.decoders.iter().enumerate() {
            let output = self.scratch.0.join(format!("decoded-{index}.jpg"));
            assert!(!output.exists(), "stale decoder output");
            let label = format!("{context}; decoder={}", decoder.display());
            // No --pixels_to_jpeg: this must use jbrd, not decode/re-encode.
            run_checked(
                Command::new(decoder)
                    .arg(input)
                    .arg(&output)
                    .arg("--num_threads=1")
                    .arg("--quiet"),
                &self.scratch.0,
                &label,
                Duration::from_secs(30),
            );
            assert_eq!(
                fs::metadata(&output).unwrap().len(),
                expected.len() as u64,
                "{label}"
            );
            let actual = fs::read(&output).unwrap();
            if let Some(i) = actual.iter().zip(expected).position(|(a, b)| a != b) {
                panic!(
                    "{label}: JPEG byte {i}: expected {}, got {}",
                    expected[i], actual[i]
                );
            }
            fs::remove_file(output).unwrap();
        }
    }
}

fn assert_png_samples(path: &Path, expected: &Image, context: &str) {
    let (color, depth, bytes, channels) = match expected.pixels() {
        PixelData::U8(values) => (png::ColorType::Rgba, png::BitDepth::Eight, values.len(), 4),
        PixelData::U16(values) => (
            png::ColorType::Grayscale,
            png::BitDepth::Sixteen,
            values.len() * 2,
            1,
        ),
        _ => panic!("unsupported acceptance fixture"),
    };
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path).unwrap()));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    let info = reader.info();
    assert_eq!(
        (info.width, info.height),
        (expected.width(), expected.height()),
        "{context}"
    );
    assert_eq!(
        (info.color_type, info.bit_depth),
        (color, depth),
        "{context}"
    );
    assert!(
        info.animation_control.is_none(),
        "unexpected animation: {context}"
    );
    assert_eq!(reader.output_buffer_size(), Some(bytes), "{context}");
    let mut actual = vec![0; bytes];
    let frame = reader.next_frame(&mut actual).unwrap();
    assert_eq!(frame.buffer_size(), bytes, "{context}");
    assert_eq!(
        (frame.color_type, frame.bit_depth),
        (color, depth),
        "{context}"
    );
    let mismatch = match expected.pixels() {
        PixelData::U8(want) => actual
            .iter()
            .zip(want)
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(i, (&a, &b))| (i, u16::from(a), u16::from(b))),
        PixelData::U16(want) => actual
            .chunks_exact(2)
            .map(|b| u16::from_be_bytes([b[0], b[1]]))
            .zip(want)
            .enumerate()
            .find(|(_, (a, b))| a != *b)
            .map(|(i, (a, &b))| (i, a, b)),
        _ => unreachable!(),
    };
    if let Some((i, got, want)) = mismatch {
        panic!(
            "{context}: ({}, {}) channel {} expected {want}, got {got}",
            i / channels % expected.width() as usize,
            i / channels / expected.width() as usize,
            i % channels
        );
    }
}

fn child_index(count: usize) -> Option<usize> {
    env::var_os(CHILD_CASE).map(|value| {
        let index: usize = value
            .to_str()
            .expect("child case UTF-8")
            .parse()
            .expect("child case index");
        assert!(index < count, "bounded child case index");
        assert!(
            rs_jxl::profile::enabled(),
            "child profiling required for dispatch proof"
        );
        index
    })
}

fn publish(bytes: &[u8]) {
    assert!(!bytes.is_empty() && bytes.len() <= OUTPUT_LIMIT);
    eprintln!(
        "{}",
        rs_jxl::profile::report().expect("child profile report")
    );
    let output = PathBuf::from(env::var_os(CHILD_OUTPUT).expect("child output path"));
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap()
        .write_all(bytes)
        .unwrap();
}

#[test]
#[ignore = "requires official djxl 0.12/0.13; cooperative MA channel/row roots"]
fn cooperative_ma_channel_roots_pixels_match_official_djxl() {
    let cases = [SERIAL, PARALLEL];
    if let Some(index) = child_index(cases.len()) {
        publish(&encode(&ma_image(), &ma_options(cases[index])).unwrap());
        return;
    }
    let oracle = Oracle::new();
    let expected = ma_image();
    for (index, case) in cases.into_iter().enumerate() {
        let context = format!("cooperative MA channel/row roots {case:?}");
        let (input, log, elapsed) = oracle.encode(
            "cooperative_ma_channel_roots_pixels_match_official_djxl",
            index,
            &context,
        );
        let _failure_log = FailureLog(&log);
        let gather = profile_rows(&log, "ma.gather_counts");
        assert_eq!(gather.len(), 1, "one shared learner: {context}\n{log}");
        assert_eq!(gather[0]["selected"], 196608);
        assert_eq!(
            gather[0]["records"], 196608,
            "unique stored records, not weights"
        );
        let roots = profile_rows(&log, "ma.channel_trees");
        assert_eq!(roots.len(), 1, "{context}\n{log}");
        assert_eq!(roots[0]["samples"], 196608);
        assert_eq!(roots[0]["channels"], 4);
        assert_eq!(roots[0]["jobs"], 16);
        assert_eq!(roots[0]["row_depth"], 2);
        assert_eq!(
            roots[0]["root_limit"],
            if case.threads == 1 { 1 } else { 2 }
        );
        assert_cooperative_dispatch(&log, case, ArenaRoute::MaChannels);
        oracle.pixels(&input, &expected, &context);
        report_case(&context, &input, &log, elapsed);
        fs::remove_file(input).unwrap();
    }
    eprintln!(
        "{} official RGBA MA-root pixel comparisons",
        2 * oracle.decoders.len()
    );
}

#[test]
#[ignore = "requires official djxl 0.12/0.13; actual detached whole-subtree learning"]
fn cooperative_ma_whole_subtrees_pixels_match_official_djxl() {
    let cases = [SERIAL, PARALLEL];
    if let Some(index) = child_index(cases.len()) {
        publish(&encode(&ma_image(), &subtree_options(cases[index])).unwrap());
        return;
    }
    let oracle = Oracle::new();
    let expected = ma_image();
    for (index, case) in cases.into_iter().enumerate() {
        let context = format!("cooperative MA whole subtrees {case:?}");
        let (input, log, elapsed) = oracle.encode(
            "cooperative_ma_whole_subtrees_pixels_match_official_djxl",
            index,
            &context,
        );
        let _failure_log = FailureLog(&log);
        let gather = profile_rows(&log, "ma.gather_counts");
        assert_eq!(gather.len(), 1, "one whole-frame learner");
        assert_eq!(gather[0]["selected"], 196608);
        assert_eq!(gather[0]["records"], 196608);
        assert!(
            profile_rows(&log, "ma.channel_trees").is_empty(),
            "channel wrapper must remain disabled"
        );
        let subtrees = profile_rows(&log, "ma.subtrees");
        if case.threads == 1 {
            assert!(subtrees.is_empty(), "null runner must not detach subtrees");
        } else {
            let dispatched: Vec<_> = subtrees
                .iter()
                .filter(|row| row.contains_key("jobs"))
                .collect();
            assert_eq!(dispatched.len(), 1, "one whole-subtree dispatch required");
            assert!(dispatched[0]["jobs"] > 1);
            assert_eq!(dispatched[0]["worker_tasks"], 2);
            assert!((1..=65536).contains(&dispatched[0]["max_samples"]));
            assert_eq!(dispatched[0]["balanced"], 0);
            assert_eq!(dispatched[0]["cache_budget_bytes"], 64 << 20);
            let added: Vec<_> = subtrees
                .iter()
                .filter_map(|row| row.get("added_nodes"))
                .collect();
            assert_eq!(added.len(), 1, "completed subtree graft required");
            assert!(*added[0] > 0, "fixture must learn nontrivial subtrees");
        }
        assert_cooperative_dispatch(&log, case, ArenaRoute::MaSubtrees);
        oracle.pixels(&input, &expected, &context);
        report_case(&context, &input, &log, elapsed);
        fs::remove_file(input).unwrap();
    }
    eprintln!(
        "{} official RGBA whole-subtree pixel comparisons",
        2 * oracle.decoders.len()
    );
}

#[test]
#[ignore = "requires official djxl 0.12/0.13; real indexed stream arena"]
fn indexed_lz_stream_arena_pixels_match_official_djxl() {
    let cases = [SERIAL, PARALLEL, QUIET];
    if let Some(index) = child_index(cases.len()) {
        publish(&encode(&lz_image(), &lz_options(cases[index])).unwrap());
        return;
    }
    let oracle = Oracle::new();
    let expected = lz_image();
    for (index, case) in cases.into_iter().enumerate() {
        let context = format!("indexed LZ stream arena {case:?}");
        let (input, log, elapsed) = oracle.encode(
            "indexed_lz_stream_arena_pixels_match_official_djxl",
            index,
            &context,
        );
        let _failure_log = FailureLog(&log);
        // Require the shared frame path; independently encoded local streams
        // cannot exercise the multi-stream arena even if options are present.
        assert!(
            log.lines()
                .any(|line| line.starts_with("lz77.parse_shared | 1 |")),
            "{context}\n{log}"
        );
        assert_cooperative_dispatch(&log, case, ArenaRoute::LzStreams);
        oracle.pixels(&input, &expected, &context);
        report_case(&context, &input, &log, elapsed);
        fs::remove_file(input).unwrap();
    }
    eprintln!(
        "{} official RGBA pixel comparisons, including hidden RGB",
        3 * oracle.decoders.len()
    );
}

#[test]
#[ignore = "requires official djxl 0.12/0.13; cooperative JPEG MA roots"]
fn cooperative_ma_jpeg_original_bytes_match_official_djxl() {
    if let Some(index) = child_index(JPEG_CASES.len() * 2) {
        let input = jpeg_fixtures::fixture(JPEG_CASES[index / 2]).unwrap();
        publish(&jpeg::transcode(&input, &jpeg_options([SERIAL, PARALLEL][index % 2])).unwrap());
        return;
    }
    let oracle = Oracle::new();
    for (fixture, name) in JPEG_CASES.into_iter().enumerate() {
        let expected = jpeg_fixtures::fixture(name).unwrap();
        for (lane, case) in [SERIAL, PARALLEL].into_iter().enumerate() {
            let context = format!("cooperative JPEG MA {name} {case:?}");
            let (input, log, elapsed) = oracle.encode(
                "cooperative_ma_jpeg_original_bytes_match_official_djxl",
                fixture * 2 + lane,
                &context,
            );
            let _failure_log = FailureLog(&log);
            let gathers = profile_rows(&log, "ma.gather_counts");
            assert!(
                gathers.len() > 1,
                "{context}: actual multiple learner domains required\n{log}"
            );
            assert!(
                gathers.iter().filter(|row| row["records"] > 0).count() > 1,
                "{context}: empty learners\n{log}"
            );
            assert_cooperative_dispatch(&log, case, ArenaRoute::JpegMa);
            oracle.original_jpeg(&input, &expected, &context);
            report_case(&context, &input, &log, elapsed);
            fs::remove_file(input).unwrap();
        }
    }
    eprintln!(
        "{} official original-JPEG byte comparisons",
        4 * oracle.decoders.len()
    );
}

#[test]
#[ignore = "requires official djxl 0.12/0.13; actual compact cooperative probe batch"]
fn cooperative_scheduler_batch_pixels_match_official_djxl() {
    let cases = [SERIAL, PARALLEL, QUIET];
    if let Some(index) = child_index(cases.len()) {
        publish(&encode_scheduler(&scheduler_image(), cases[index]));
        return;
    }
    let oracle = Oracle::new();
    let expected = scheduler_image();
    for (index, case) in cases.into_iter().enumerate() {
        let context = format!("compact scheduler cooperative batch {case:?}");
        let (input, log, elapsed) = oracle.encode(
            "cooperative_scheduler_batch_pixels_match_official_djxl",
            index,
            &context,
        );
        let _failure_log = FailureLog(&log);
        assert!(
            log.contains("[parallel acceptance] scheduler:"),
            "{context}: missing native route proof\n{log}"
        );
        assert_cooperative_dispatch(&log, case, ArenaRoute::Scheduler);
        oracle.pixels(&input, &expected, &context);
        report_case(&context, &input, &log, elapsed);
        fs::remove_file(input).unwrap();
    }
    eprintln!(
        "{} official Gray16 scheduler pixel comparisons",
        3 * oracle.decoders.len()
    );
}

#[test]
fn fixtures_and_policies_reach_source_parallel_gates_with_bounded_inputs() {
    let ma = ma_image();
    assert_eq!(ma.bits(), 8);
    assert_eq!(ma.layout(), PixelLayout::Rgba);
    for case in [SERIAL, PARALLEL] {
        let options = ma_options(case);
        options.modular.validate().unwrap();
        let policy = options.modular.sampling.as_ref().unwrap();
        assert_eq!(policy.nb_repeats, 1.0);
        assert_eq!(policy.max_property_values, 256);
        for property in 0..=3 {
            assert!(policy.properties.contains(&property));
        }
        assert_eq!(options.modular.sample_limit, 196608);
        assert!(!options.modular.source_ma_parallel.subtrees);
        let subtrees = subtree_options(case);
        subtrees.modular.validate().unwrap();
        assert_eq!(
            subtrees.modular.source_channel_tree,
            ChannelTreeMode::Disabled
        );
        assert!(subtrees.modular.source_ma_parallel.subtrees);
        assert!(!subtrees.modular.source_ma_parallel.small_subtrees);
        assert!(!subtrees.modular.source_ma_parallel.balanced);
    }
    let image = lz_image();
    let PixelData::U8(samples) = image.pixels() else {
        panic!("RGBA8 fixture");
    };
    assert!(
        samples
            .chunks_exact(4)
            .filter(|p| p[3] == 0 && p[..3] != [0, 0, 0])
            .count()
            > 100
    );
    assert_eq!((image.width(), image.height()), (384, 256));
    let streams = [256 * 256 * 4, 128 * 256 * 4];
    assert!(streams.iter().all(|&tokens| tokens >= 65536));
    assert_eq!(streams.iter().sum::<usize>(), samples.len());
    for case in [SERIAL, PARALLEL, QUIET] {
        let options = lz_options(case);
        options.modular.validate().unwrap();
        let config = options.modular.source_lz77.unwrap();
        assert!(
            config.stream_arena && config.parallel_index_build && config.require_global_savings
        );
        assert!(!config.shared_index && !config.small_index);
        assert_eq!(config.index_mode, SourceIndexMode::All);
        assert_eq!(config.match_cost, SourceMatchCost::Frontier);
        assert_eq!(config.block_mode, SourceBlockMode::Intervals);
    }
    let search = scheduler_image();
    let pixels = u64::from(search.width()) * u64::from(search.height());
    assert!(pixels > 1 << 19 && pixels < 1 << 20);
    assert_eq!(search.bits(), 16);
}

#[test]
fn activation_profile_parser_ignores_timing_rows_and_keeps_all_dispatches() {
    let log = "[rs-jxl profile] ma.gather_counts: visited=24 selected=24 records=20\n\
        [rs-jxl profile] lz77.parse_shared: 12.5 ms\n\
        [rs-jxl profile] ma.gather_counts: visited=32 selected=32 records=28\n";
    let rows = profile_rows(log, "ma.gather_counts");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["records"], 20);
    assert_eq!(rows[1]["records"], 28);
    assert!(profile_rows(log, "absent.dispatch").is_empty());
    let rows = profile_rows(
        "[rs-jxl profile] ma.subtrees: jobs=4 worker_tasks=2 balanced=false\n\
         [rs-jxl profile] ma.subtrees: added_nodes=48\n\
         [rs-jxl profile] ma.subtrees: jobs=8 worker_tasks=2 balanced=true\n",
        "ma.subtrees",
    );
    assert_eq!(rows[0]["balanced"], 0);
    assert_eq!(rows[1]["added_nodes"], 48);
    assert_eq!(rows[2]["balanced"], 1);
}

#[test]
fn cooperative_activation_checks_actual_entry_caps_and_inherited_nested_policy() {
    for (route, jobs, workers, roots) in [
        (ArenaRoute::MaChannels, 16, 4, 2),
        (ArenaRoute::MaSubtrees, 5, 4, 2),
        (ArenaRoute::LzStreams, 2, 4, 2),
        (ArenaRoute::JpegMa, 3, 4, 2),
        (ArenaRoute::Scheduler, 7, 3, 3),
    ] {
        for case in [PARALLEL, QUIET] {
            let quiet = usize::from(case.quiet);
            let log = format!(
                "[rs-jxl profile] execution.cooperative: jobs={jobs} workers={workers} root_limit={roots} quiet={quiet} nested=0\n\
                 [rs-jxl profile] execution.cooperative: jobs=9 workers={workers} root_limit=1 quiet={quiet} nested=1\n"
            );
            assert_cooperative_dispatch(&log, case, route);
        }
        if route != ArenaRoute::Scheduler {
            assert_cooperative_dispatch("", SERIAL, route);
        }
    }
    assert_cooperative_dispatch(
        "[rs-jxl profile] execution.cooperative: jobs=7 workers=1 root_limit=3 quiet=0 nested=0\n",
        SERIAL,
        ArenaRoute::Scheduler,
    );
}

#[test]
fn cooperative_activation_rejects_bypasses_wrong_caps_and_nested_only_evidence() {
    let valid =
        "[rs-jxl profile] execution.cooperative: jobs=2 workers=4 root_limit=2 quiet=0 nested=0\n";
    let invalid = [
        String::new(),
        valid.replace("workers=4", "workers=1"),
        valid.replace("root_limit=2", "root_limit=1"),
        valid.replace("quiet=0", "quiet=1"),
        valid.replace("nested=0", "nested=1"),
        valid.replace("jobs=2", "jobs=1"),
    ];
    for log in invalid {
        assert!(
            std::panic::catch_unwind(|| {
                assert_cooperative_dispatch(&log, PARALLEL, ArenaRoute::LzStreams);
            })
            .is_err(),
            "accepted invalid activation evidence: {log}"
        );
    }
    assert!(
        std::panic::catch_unwind(|| {
            assert_cooperative_dispatch(valid, SERIAL, ArenaRoute::LzStreams);
        })
        .is_err(),
        "a null runner must not enter an arena"
    );
}
