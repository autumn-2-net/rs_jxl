//! Official pixel oracle for frozen enc_ma.cc::FindBestSplit channel/row trees.
//! Eight encodes, two small RGBA fixtures; no compressed-byte/hash requirement.
//! Parent runs centrally with RS_JXL_DJXL (an OS path list, ';' on Windows):
//! cargo test -p rs-jxl --test channel_tree_reference -- --ignored --nocapture
//!
//! Oracle setup/comparison follows reference.rs. Its helpers are private, so
//! this independently owned test retains the same conventions locally. Each
//! encode re-enters this test binary with profiling enabled: this captures the
//! real prepared-record/job counts without unsafe process-global env mutation,
//! private library hooks, or a subprocess/native encoder implementation.

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::modular::{
    ma_train::{ChannelTreeMode, MAX_LEAVES},
    predictor::Predictor,
    sample::{SamplingPolicy, SourcePolicy},
};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode};

const TEST_NAME: &str = "source_channel_tree_modes_match_official_djxl";
const CHILD_CASE: &str = "RS_JXL_CHANNEL_TREE_TEST_CHILD";
const CHILD_OUTPUT: &str = "RS_JXL_CHANNEL_TREE_TEST_OUTPUT";
const OUTPUT_LIMIT: usize = 4 << 20;
const LOG_LIMIT: u64 = 8 << 20;
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug)]
struct Case {
    width: u32,
    height: u32,
    mode: ChannelTreeMode,
    threads: usize,
    jobs: usize,
    row_depth: usize,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (mode, jobs, row_depth) in [
        (ChannelTreeMode::Channels, 4, 0),
        (ChannelTreeMode::ChannelsAndRows, 8, 1),
        (ChannelTreeMode::ChannelsAndRowsTwice, 16, 2),
    ] {
        for threads in [1, 4] {
            cases.push(Case {
                width: 192,
                height: 256,
                mode,
                threads,
                jobs,
                row_depth,
            });
        }
    }
    for threads in [1, 4] {
        cases.push(Case {
            width: 7,
            height: 2048,
            mode: ChannelTreeMode::Channels,
            threads,
            jobs: 4,
            row_depth: 0,
        });
    }
    cases
}

fn fixture(case: Case) -> Image {
    let mut state = 0x79f5_6d2bu32;
    let mut samples = Vec::with_capacity(case.width as usize * case.height as usize * 4);
    for y in 0..case.height {
        for x in 0..case.width {
            for channel in 0..4 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let value = match channel {
                    1 if y < 64 => (x + y * 3) as u8,
                    2 if (128..192).contains(&y) => (x * 5 + y) as u8,
                    3 if (y * case.width + x).is_multiple_of(17) => 0,
                    _ => state as u8,
                };
                samples.push(value);
            }
        }
    }
    samples[..4].copy_from_slice(&[17, 223, 91, 0]);
    assert!(
        samples
            .chunks_exact(4)
            .filter(|p| p[3] == 0 && p[..3] != [0, 0, 0])
            .count()
            > 100
    );
    Image::new(
        case.width,
        case.height,
        PixelLayout::Rgba,
        PixelData::U8(samples),
    )
    .unwrap()
}

fn options(case: Case) -> EncodeOptions {
    let policy = SamplingPolicy::from_source(&SourcePolicy {
        speed_tier: 1,
        predictor: Predictor::Gradient,
        learning_fraction: 1.0,
        num_extra_channels: 1,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(policy.nb_repeats, 1.0);
    assert_eq!(policy.max_property_values, 256);
    assert_eq!(policy.properties.first(), Some(&0));
    assert!(
        policy.properties.contains(&1)
            && policy.properties.contains(&2)
            && policy.properties.contains(&3)
    );
    EncodeOptions {
        threads: case.threads,
        shared_tree: true,
        max_pixels: u64::from(case.width) * u64::from(case.height),
        max_output_bytes: OUTPUT_LIMIT,
        modular: ModularOptions {
            predictor: Predictor::Gradient,
            rct: None,
            lz77: false,
            sampling: Some(policy),
            sample_limit: case.width as usize * case.height as usize * 4,
            max_tree_leaves: MAX_LEAVES,
            source_channel_tree: case.mode,
            source_channel_tree_final_only: false,
            source_tree_entropy: true,
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
                    let (name, value) = field.split_once('=').expect("profile key=value");
                    (name, value.parse().expect("integer profile field"))
                })
                .collect()
        })
        .collect()
}

fn assert_active_wrapper(log: &str, case: Case) {
    let selected = case.width as usize * case.height as usize * 4;
    let gather = profile_rows(log, "ma.gather_counts");
    assert_eq!(
        gather.len(),
        1,
        "expected one shared learner: {case:?}\n{log}"
    );
    assert_eq!(
        gather[0]["selected"], selected,
        "fraction must sample every pixel"
    );
    assert_eq!(gather[0]["visited"], selected);
    assert_eq!(gather[0]["variable_channels"], 4);
    // Channel/stream/x/y distinguish every pixel, including narrow groups.
    // With 256 coordinate bins the wide fixture has 49,152 records/channel,
    // 24,576 in each first row half, and 12,288 in each second row half.
    assert_eq!(
        gather[0]["records"], selected,
        "stored records, not weighted sample mass"
    );
    let wrapper = profile_rows(log, "ma.channel_trees");
    assert_eq!(
        wrapper.len(),
        1,
        "requested channel wrapper did not run: {case:?}\n{log}"
    );
    assert_eq!(wrapper[0]["samples"], selected);
    assert!(wrapper[0]["samples"] >= 8192);
    assert_eq!(wrapper[0]["channels"], 4);
    assert_eq!(
        wrapper[0]["jobs"], case.jobs,
        "row partition actually taken, not just requested"
    );
    assert_eq!(wrapper[0]["row_depth"], case.row_depth);
    if case.row_depth != 0 {
        assert!(selected / 4 >= 16384);
        if case.row_depth == 2 {
            assert!(selected / 8 >= 16384);
        }
    }
    eprintln!(
        "{case:?}: selected={selected}, stored={}, channels=4, jobs={}, row_depth={}",
        wrapper[0]["samples"], wrapper[0]["jobs"], wrapper[0]["row_depth"]
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
                "rs-jxl-channel-tree-{}-{stamp}-{id}",
                std::process::id()
            ));
            assert_eq!(path.parent(), Some(root.as_path()));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        }
        panic!("cannot create unique channel-tree oracle directory");
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
            .unwrap_or_else(|error| panic!("{context}: spawn: {error}")),
    );
    // Release Command's file handles before eventual Windows scratch cleanup.
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

fn decoder_paths(scratch: &Path) -> Vec<PathBuf> {
    let value = env::var_os("RS_JXL_DJXL")
        .expect("explicit oracle test requires RS_JXL_DJXL official decoder path list");
    let paths: Vec<_> = env::split_paths(&value)
        .map(|path| {
            assert!(
                !path.as_os_str().is_empty() && path.is_file(),
                "invalid official decoder path: {}",
                path.display()
            );
            path.canonicalize().unwrap()
        })
        .collect();
    assert!(!paths.is_empty(), "RS_JXL_DJXL contains no decoder paths");
    for path in &paths {
        let banner = run_checked(
            Command::new(path).arg("--version"),
            scratch,
            "decoder version",
            Duration::from_secs(30),
        );
        assert!(
            banner.contains("djxl v"),
            "not an official djxl banner: {banner}"
        );
        eprintln!("{}: {}", path.display(), banner.trim());
    }
    paths
}

fn assert_png_samples(path: &Path, expected: &Image, context: &str) {
    let PixelData::U8(samples) = expected.pixels() else {
        panic!("8-bit fixture expected");
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
        (png::ColorType::Rgba, png::BitDepth::Eight),
        "{context}"
    );
    assert!(
        info.animation_control.is_none(),
        "unexpected animation: {context}"
    );
    assert_eq!(
        reader.output_buffer_size(),
        Some(samples.len()),
        "{context}"
    );
    let mut actual = vec![0; samples.len()];
    let output = reader.next_frame(&mut actual).unwrap();
    assert_eq!(output.buffer_size(), samples.len(), "{context}");
    assert_eq!(
        (output.color_type, output.bit_depth),
        (png::ColorType::Rgba, png::BitDepth::Eight),
        "{context}"
    );
    // No alpha masking: compare all hidden RGB and alpha samples as well.
    if let Some((i, (&got, &want))) = actual
        .iter()
        .zip(samples)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        panic!(
            "{context}: ({}, {}) channel {} expected {want}, got {got}",
            i / 4 % expected.width() as usize,
            i / 4 / expected.width() as usize,
            i % 4
        );
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; active source channel/row partitions"]
fn source_channel_tree_modes_match_official_djxl() {
    let cases = cases();
    if let Some(index) = env::var_os(CHILD_CASE) {
        let index: usize = index
            .to_str()
            .expect("child case UTF-8")
            .parse()
            .expect("child case index");
        let case = *cases.get(index).expect("bounded child case index");
        assert!(
            rs_jxl::profile::enabled(),
            "child profiling is required for activation proof"
        );
        let output = PathBuf::from(env::var_os(CHILD_OUTPUT).expect("child output path"));
        let bytes = encode(&fixture(case), &options(case))
            .unwrap_or_else(|error| panic!("{case:?}: {error}"));
        assert!(!bytes.is_empty() && bytes.len() <= OUTPUT_LIMIT);
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        return;
    }
    let scratch = ScratchDir::new();
    let decoders = decoder_paths(&scratch.0);
    let executable = env::current_exe().expect("integration test executable");
    let mut checks = 0;
    for (index, case) in cases.into_iter().enumerate() {
        let label = format!("channel tree {case:?}");
        let input = scratch.0.join(format!("case-{index}.jxl"));
        let log = run_checked(
            Command::new(&executable)
                .args([
                    "--exact",
                    TEST_NAME,
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_CASE, index.to_string())
                .env(CHILD_OUTPUT, &input)
                .env("RS_JXL_PROFILE", "1"),
            &scratch.0,
            &label,
            Duration::from_secs(180),
        );
        assert_active_wrapper(&log, case);
        assert!(fs::metadata(&input).unwrap().len() <= OUTPUT_LIMIT as u64);
        let expected = fixture(case);
        for (decoder_index, decoder) in decoders.iter().enumerate() {
            let output = scratch
                .0
                .join(format!("case-{index}-decoder-{decoder_index}.png"));
            assert!(!output.exists(), "stale decoder output");
            let context = format!("{label}; decoder={}", decoder.display());
            run_checked(
                Command::new(decoder)
                    .arg(&input)
                    .arg(&output)
                    .arg("--bits_per_sample=8")
                    .arg("--num_threads=1")
                    .arg("--quiet"),
                &scratch.0,
                &context,
                Duration::from_secs(30),
            );
            assert_png_samples(&output, &expected, &context);
            fs::remove_file(output).unwrap();
            checks += 1;
        }
        fs::remove_file(input).unwrap();
    }
    eprintln!("{checks} official pixel comparisons; eight active channel-tree encodes");
}

#[test]
fn channel_tree_fixture_budgets_cover_both_row_levels_without_large_inputs() {
    let cases = cases();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let image = fixture(case);
        let options = options(case);
        options.modular.validate().unwrap();
        assert_eq!(image.bits(), 8);
        assert_eq!(image.layout(), PixelLayout::Rgba);
        assert!(options.max_pixels <= 49152);
        assert_eq!(options.modular.sample_limit as u64, 4 * options.max_pixels);
        assert!(options.modular.sample_limit >= 8192);
        if case.row_depth != 0 {
            assert_eq!(options.modular.sample_limit, 196608);
            assert_eq!(case.jobs, 4 << case.row_depth);
            assert!(options.modular.sample_limit / 8 >= 16384);
        }
    }
}

#[test]
fn profile_activation_parser_checks_stored_counts_and_real_row_jobs() {
    let case = cases()[4];
    let log = "[rs-jxl profile] ma.gather_counts: visited=196608 selected=196608 records=196608 source_total_pixels=196608 variable_channels=4\n\
        [rs-jxl profile] ma.channel_trees: channels=4 jobs=16 samples=196608 row_depth=2 concurrent_roots=1\n";
    assert_active_wrapper(log, case);
    assert!(profile_rows("", "ma.channel_trees").is_empty());
    assert_eq!(profile_rows(log, "ma.channel_trees")[0]["jobs"], 16);
}
