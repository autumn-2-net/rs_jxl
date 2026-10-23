//! Development-only independent decoder acceptance for JPEG transcoding.
//! The coefficient backend stays private; adapter cases exercise the byte API.
//! RS_JXL_DJXL is an OS path list of official decoders (e.g. 0.12 and 0.13).
//! RS_JXL_JPEG_INPUTS is an OS path list of JPEG files/nonrecursive directories;
//! select 1..=17 distinct files. Missing configuration or exceeded caps fail.
//! RS_JXL_JPEG_MATRIX=0 (or unset) selects three configurations: source defaults,
//! Falcon/Default/Disabled and Tortoise/Adaptive/Enabled. A value of 1 selects
//! all 12 combinations of those tiers, three order policies and both CfL modes.
//! No official encoder is invoked. Timing covers encode_coefficients only,
//! excluding file reads, coefficient extraction, entropy erasure and decoding.
//! The separate synthetic test uses shared in-memory fixtures and needs only
//! RS_JXL_DJXL; RS_JXL_JPEG_MATRIX optionally expands its configurations too.
//! Targeted tier, execution, metadata and byte-adapter tests use fixed small
//! schedules, independent of that matrix variable. No input paths are required.

#[path = "../../tests/support/jpeg_fixtures.rs"]
mod fixtures;

use std::{
    collections::BTreeSet,
    env,
    ffi::OsStr,
    fs::{self, File, Metadata, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    entropy::options::SpeedTier,
    jpeg::{
        archival::{self, TranscodeLimits, TranscodeOptions},
        cfl::CflMode,
        coefficient_order::OrderPolicy,
        coefficient_reader::{CoefficientReadLimits, read_coefficients_with_limits},
        data::{CodingProcess, JpegData, JpegRecord},
        reader::{ReadLimits, read_jpeg_with_limits},
        transcode::{Options, encode_coefficients},
    },
};

const MAX_INPUT: usize = 16 << 20;
const MAX_OUTPUT: usize = 64 << 20;
const MAX_COEFFICIENT_BYTES: usize = 256 << 20;
const MAX_PIXELS: u64 = 64_000_000;
const MAX_LOG: usize = 64 << 10;
const MAX_FIXTURES: usize = 17;
const MAX_DISCOVERY: usize = 4096;
const TIMEOUT: Duration = Duration::from_secs(60);
const TEMP_PREFIX: &str = "rs-jxl-jpeg-transcode-";
const FILES: [&str; 3] = ["encoded.jxl", "decoded.jpg", "stderr.txt"];
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn configurations(matrix: Option<&OsStr>) -> Result<Vec<Options>, &'static str> {
    let full = match matrix {
        None => false,
        Some(value) if value == "0" => false,
        Some(value) if value == "1" => true,
        Some(_) => return Err("RS_JXL_JPEG_MATRIX must be 0 or 1"),
    };
    let bounded = Options {
        max_output_bytes: MAX_OUTPUT,
        max_coefficients: MAX_COEFFICIENT_BYTES / std::mem::size_of::<i32>(),
        max_tokens: 64 << 20,
        ..Options::default()
    };
    if !full {
        return Ok(vec![
            bounded.clone(),
            Options {
                tier: SpeedTier::Falcon,
                order: OrderPolicy::Default,
                cfl: CflMode::Disabled,
                ..bounded.clone()
            },
            Options {
                tier: SpeedTier::Tortoise,
                order: OrderPolicy::Adaptive,
                cfl: CflMode::Enabled,
                ..bounded
            },
        ]);
    }
    let mut options = Vec::with_capacity(12);
    for tier in [SpeedTier::Falcon, SpeedTier::Tortoise] {
        for order in [
            OrderPolicy::Source,
            OrderPolicy::Default,
            OrderPolicy::Adaptive,
        ] {
            for cfl in [CflMode::Disabled, CflMode::Enabled] {
                options.push(Options {
                    tier,
                    order,
                    cfl,
                    ..bounded.clone()
                });
            }
        }
    }
    Ok(options)
}

fn tier_configurations() -> Vec<Options> {
    let base = configurations(None).unwrap().remove(0);
    (-1i8..=9)
        .map(|tier| Options {
            tier: SpeedTier::try_from(tier).unwrap(),
            order: OrderPolicy::Source,
            cfl: CflMode::Enabled,
            ..base.clone()
        })
        .collect()
}

fn execution_configurations() -> Vec<Options> {
    let base = configurations(None).unwrap().remove(0);
    [
        (SpeedTier::Falcon, 1, 0),
        (SpeedTier::Falcon, 2, 1),
        (SpeedTier::Tortoise, 1, 4),
        (SpeedTier::Tortoise, 2, 0),
    ]
    .into_iter()
    .map(|(tier, threads, decoding_speed)| Options {
        tier,
        threads,
        decoding_speed,
        ..base.clone()
    })
    .collect()
}

fn adapter_configurations() -> Vec<TranscodeOptions> {
    let limits = TranscodeLimits {
        input: CoefficientReadLimits {
            structural: ReadLimits {
                max_input_bytes: MAX_INPUT,
                ..ReadLimits::default()
            },
            max_coefficient_bytes: MAX_COEFFICIENT_BYTES,
            ..CoefficientReadLimits::default()
        },
        max_output_bytes: MAX_OUTPUT,
        max_coefficients: MAX_COEFFICIENT_BYTES / std::mem::size_of::<i32>(),
        max_tokens: 64 << 20,
        sample_limit: 1 << 20,
    };
    vec![
        TranscodeOptions {
            effort: 1,
            threads: 1,
            cooperative_ma: None,
            decoding_speed: 4,
            order: OrderPolicy::Default,
            cfl: CflMode::Disabled,
            reconstruction_quality: Some(0),
            limits: limits.clone(),
        },
        TranscodeOptions {
            effort: 3,
            threads: 2,
            cooperative_ma: None,
            decoding_speed: 1,
            order: OrderPolicy::Source,
            cfl: CflMode::Enabled,
            reconstruction_quality: None,
            limits: limits.clone(),
        },
        TranscodeOptions {
            effort: 11,
            threads: 1,
            cooperative_ma: None,
            decoding_speed: 0,
            order: OrderPolicy::Adaptive,
            cfl: CflMode::Enabled,
            reconstruction_quality: Some(11),
            limits,
        },
    ]
}

fn is_link(metadata: &Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn is_jpeg(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg"))
}

fn discover(roots: &OsStr) -> Vec<PathBuf> {
    let mut files = BTreeSet::new();
    let mut inspected = 0usize;
    let mut excluded = 0usize;
    let mut count_entry = || {
        inspected += 1;
        assert!(
            inspected <= MAX_DISCOVERY,
            "nonrecursive discovery exceeds {MAX_DISCOVERY} entries"
        );
    };
    let mut add = |path: &Path| {
        let metadata = fs::symlink_metadata(path).expect("inspect JPEG fixture");
        assert!(
            metadata.is_file() && !is_link(&metadata) && is_jpeg(path),
            "expected regular, unlinked JPEG: {}",
            path.display()
        );
        files.insert(path.canonicalize().expect("resolve JPEG fixture"));
        assert!(
            files.len() <= MAX_FIXTURES,
            "more than {MAX_FIXTURES} JPEGs selected; narrow RS_JXL_JPEG_INPUTS (no truncation)"
        );
    };
    for root in env::split_paths(roots) {
        count_entry();
        assert!(!root.as_os_str().is_empty(), "empty JPEG input path");
        let metadata = fs::symlink_metadata(&root)
            .unwrap_or_else(|error| panic!("inspect {}: {error}", root.display()));
        assert!(
            !is_link(&metadata),
            "linked JPEG input root: {}",
            root.display()
        );
        if metadata.is_dir() {
            for entry in fs::read_dir(&root).expect("read JPEG input directory") {
                count_entry();
                let entry = entry.expect("read JPEG directory entry");
                let path = entry.path();
                if entry.file_type().expect("inspect JPEG entry").is_dir() || !is_jpeg(&path) {
                    excluded += 1;
                    continue;
                }
                add(&path);
            }
        } else {
            add(&root);
        }
    }
    assert!(
        !files.is_empty(),
        "RS_JXL_JPEG_INPUTS selected no JPEG files"
    );
    eprintln!(
        "selected {} JPEGs; {excluded} non-JPEG entries/subdirectories excluded (nonrecursive)",
        files.len()
    );
    files.into_iter().collect()
}

fn decoders() -> Vec<PathBuf> {
    let value =
        env::var_os("RS_JXL_DJXL").expect("RS_JXL_DJXL official decoder OS path list required");
    let mut paths = Vec::new();
    for path in env::split_paths(&value) {
        assert!(!path.as_os_str().is_empty(), "empty decoder path");
        assert!(paths.len() < 4, "select at most four official decoders");
        let path = path
            .canonicalize()
            .expect("resolve official decoder executable");
        assert!(path.is_file(), "decoder is not a file: {}", path.display());
        assert!(
            !paths.contains(&path),
            "duplicate decoder: {}",
            path.display()
        );
        paths.push(path);
    }
    assert!(!paths.is_empty(), "select at least one official decoder");
    paths
}

fn read_bounded(path: &Path, maximum: usize) -> io::Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || is_link(&metadata) || metadata.len() > maximum as u64 {
        return Err(io::Error::other(
            "file is linked, nonregular or exceeds byte cap",
        ));
    }
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > maximum as u64 {
        return Err(io::Error::other(
            "opened file exceeds byte cap or is nonregular",
        ));
    }
    let length = usize::try_from(metadata.len()).map_err(io::Error::other)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(io::Error::other)?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)?;
    if file.read(&mut [0])? != 0 {
        return Err(io::Error::other("file grew while reading"));
    }
    Ok(bytes)
}

struct Scratch {
    root: PathBuf,
    path: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        let root = env::temp_dir().canonicalize().expect("resolve temp root");
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        for _ in 0..100 {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!("{TEMP_PREFIX}{}-{stamp}-{id}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self { root, path },
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        }
        panic!("cannot create unique JPEG oracle scratch directory");
    }

    fn check_directory(&self) -> io::Result<()> {
        let metadata = fs::symlink_metadata(&self.path)?;
        if !metadata.is_dir()
            || is_link(&metadata)
            || self.path.parent() != Some(self.root.as_path())
            || self.path.canonicalize()? != self.path
            || self.root.canonicalize()? != self.root
            || !self
                .path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with(TEMP_PREFIX))
        {
            return Err(io::Error::other("changed/out-of-root scratch directory"));
        }
        Ok(())
    }

    fn remove(&self, name: &str) -> io::Result<()> {
        self.check_directory()?;
        if !FILES.contains(&name) {
            return Err(io::Error::other("unowned scratch file"));
        }
        let path = self.path.join(name);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        if !metadata.is_file() || is_link(&metadata) || path.canonicalize()? != path {
            return Err(io::Error::other("changed scratch file"));
        }
        fs::remove_file(path)
    }

    fn cleanup(&self) -> io::Result<()> {
        self.check_directory()?;
        // Never recurse or remove files created under unexpected names.
        for name in FILES {
            self.remove(name)?;
        }
        fs::remove_dir(&self.path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = self.cleanup() {
            if thread::panicking() {
                eprintln!("scratch cleanup {}: {error}", self.path.display());
            } else {
                panic!("scratch cleanup {}: {error}", self.path.display());
            }
        }
    }
}

struct Reap(Child);
impl Drop for Reap {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_)))
            && let Err(error) = self.0.kill()
        {
            eprintln!("kill official decoder: {error}");
        }
        if let Err(error) = self.0.wait() {
            if thread::panicking() {
                eprintln!("reap official decoder: {error}");
            } else {
                panic!("reap official decoder: {error}");
            }
        }
    }
}

fn check_size(path: &Path, maximum: usize) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata)
            if metadata.is_file() && !is_link(&metadata) && metadata.len() <= maximum as u64 =>
        {
            Ok(true)
        }
        Ok(_) => Err(io::Error::other(
            "oracle output is linked, nonregular or exceeds byte cap",
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn decode(decoder: &Path, scratch: &Scratch) -> Vec<u8> {
    scratch
        .check_directory()
        .expect("check decoder scratch directory");
    let output = scratch.path.join("decoded.jpg");
    let log = scratch.path.join("stderr.txt");
    assert!(
        !check_size(&output, MAX_INPUT).expect("check fresh decoder output path"),
        "stale decoder output"
    );
    let log_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&log)
        .expect("create decoder log");
    let mut command = Command::new(decoder);
    command
        .arg(scratch.path.join("encoded.jxl"))
        .arg(&output)
        .args(["--num_threads=1", "--quiet"])
        .current_dir(&scratch.path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log_file);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW.
    }
    let mut child = Reap(
        command
            .spawn()
            .unwrap_or_else(|error| panic!("spawn {}: {error}", decoder.display())),
    );
    drop(command); // Close the parent's stderr handle before Windows cleanup.
    let start = Instant::now();
    let status = loop {
        check_size(&output, MAX_INPUT).expect("bounded decoder output");
        assert!(
            check_size(&log, MAX_LOG).expect("bounded decoder stderr"),
            "missing decoder log"
        );
        if let Some(status) = child.0.try_wait().expect("poll official decoder") {
            break status;
        }
        assert!(
            start.elapsed() < TIMEOUT,
            "{} exceeded {TIMEOUT:?}; killing/reaping",
            decoder.display()
        );
        thread::sleep(Duration::from_millis(20));
    };
    drop(child);
    let log_bytes = read_bounded(&log, MAX_LOG).expect("read bounded decoder stderr");
    assert!(
        status.success(),
        "{}: {status}: {}",
        decoder.display(),
        String::from_utf8_lossy(&log_bytes)
    );
    let bytes = read_bounded(&output, MAX_INPUT).expect("read bounded reconstructed JPEG");
    scratch.remove("decoded.jpg").expect("remove decoded JPEG");
    scratch.remove("stderr.txt").expect("remove decoder log");
    bytes
}

fn extract_without_entropy(original: &[u8], context: &Path) -> JpegData {
    let structural = ReadLimits {
        max_input_bytes: MAX_INPUT,
        ..ReadLimits::default()
    };
    {
        let header = read_jpeg_with_limits(original, &structural)
            .unwrap_or_else(|error| panic!("{}: structural JPEG read: {error}", context.display()));
        let pixels = u64::from(header.width) * u64::from(header.height);
        assert!(
            (1..=MAX_PIXELS).contains(&pixels),
            "{}: pixel cap {MAX_PIXELS}, got {pixels}",
            context.display()
        );
    }
    let limits = CoefficientReadLimits {
        structural,
        max_coefficient_bytes: MAX_COEFFICIENT_BYTES,
        ..CoefficientReadLimits::default()
    };
    let mut jpeg = read_coefficients_with_limits(original, &limits)
        .unwrap_or_else(|error| panic!("{}: coefficient read: {error}", context.display()));
    assert!(
        jpeg.reconstruct().expect("reconstruct input archive") == original,
        "{}: input archive mismatch",
        context.display()
    );
    let mut scans = 0;
    for record in &mut jpeg.records {
        if let JpegRecord::Entropy(entropy) = record {
            scans += 1;
            entropy.bytes.clear();
            entropy.bytes.shrink_to_fit();
        }
    }
    assert!(scans > 0, "no JPEG entropy records");
    assert!(jpeg.records.iter().all(
        |record| !matches!(record, JpegRecord::Entropy(entropy) if !entropy.bytes.is_empty())
    ));
    jpeg
}

fn first_difference(original: &[u8], restored: &[u8]) -> Option<usize> {
    original
        .iter()
        .zip(restored)
        .position(|(a, b)| a != b)
        .or_else(|| {
            (original.len() != restored.len()).then_some(original.len().min(restored.len()))
        })
}

#[test]
#[ignore = "explicit RS_JXL_DJXL and RS_JXL_JPEG_INPUTS; private Rust coefficient-frame acceptance"]
fn rust_coefficient_transcode_restores_real_jpegs_with_official_decoders() {
    let decoders = decoders();
    let configs = configurations(env::var_os("RS_JXL_JPEG_MATRIX").as_deref())
        .expect("JPEG config selection");
    let inputs = discover(
        &env::var_os("RS_JXL_JPEG_INPUTS")
            .expect("explicit JPEG file/directory OS path list required"),
    );
    let mut checks = 0usize;
    let mut progressive = 0usize;
    let mut restarted = 0usize;
    let mut encoded_pixels = 0u64;
    let mut encoded_bytes = 0u64;
    let mut encode_seconds = 0.0;
    for path in &inputs {
        let original = read_bounded(path, MAX_INPUT)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let jpeg = extract_without_entropy(&original, path);
        let has_restart = jpeg.records.iter().any(|record| matches!(record, JpegRecord::Entropy(entropy) if !entropy.restart_markers.is_empty()));
        progressive += usize::from(jpeg.coding_process == Some(CodingProcess::Progressive));
        restarted += usize::from(has_restart);
        let pixels = u64::from(jpeg.width) * u64::from(jpeg.height);
        let scratch = Scratch::new();
        for options in &configs {
            let label = format!("{:?}/{:?}/{:?}", options.tier, options.order, options.cfl);
            eprintln!(
                "encoding {} {}x{} {:?}, {} scans, restarts={has_restart}, {label}; archived entropy erased",
                path.display(),
                jpeg.width,
                jpeg.height,
                jpeg.coding_process,
                jpeg.scan_info.len()
            );
            let start = Instant::now();
            let result = encode_coefficients(&jpeg, options);
            let seconds = start.elapsed().as_secs_f64();
            let encoded = result.unwrap_or_else(|error| {
                panic!(
                    "{} {label}: Rust coefficient encode: {error}",
                    path.display()
                )
            });
            assert!(
                !encoded.is_empty() && encoded.len() <= MAX_OUTPUT,
                "Rust output cap"
            );
            let bpp = encoded.len() as f64 * 8.0 / pixels as f64;
            eprintln!(
                "{} {label}: JXL={} bytes, {bpp:.6} bpp, Rust coeffencode={seconds:.6}s, {:.3} MP/s (excludes read/extraction/decoder)",
                path.display(),
                encoded.len(),
                pixels as f64 / 1_000_000.0 / seconds
            );
            encoded_pixels += pixels;
            encoded_bytes += encoded.len() as u64;
            encode_seconds += seconds;
            {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(scratch.path.join("encoded.jxl"))
                    .expect("create Rust JXL input");
                file.write_all(&encoded).expect("write Rust JXL input");
                file.flush().expect("flush Rust JXL input");
            }
            drop(encoded);
            for decoder in &decoders {
                let restored = decode(decoder, &scratch);
                let difference = first_difference(&original, &restored);
                assert!(
                    difference.is_none(),
                    "{} {label} decoder={}: first mismatch={difference:?}, lengths={}/{}",
                    path.display(),
                    decoder.display(),
                    original.len(),
                    restored.len()
                );
                checks += 1;
                eprintln!(
                    "{} {label}: exact {} JPEG bytes via {}",
                    path.display(),
                    original.len(),
                    decoder.display()
                );
            }
            scratch
                .remove("encoded.jxl")
                .expect("remove Rust JXL input");
        }
    }
    assert_eq!(checks, inputs.len() * configs.len() * decoders.len());
    eprintln!(
        "{} JPEGs, {progressive} progressive, {restarted} with restarts, {} configurations, {} decoders: {checks} byte-exact checks; all frames Rust-produced with archived entropy erased. Total JXL={encoded_bytes} bytes, Rust coeffencode={encode_seconds:.6}s, {:.3} MP/s (excludes reads/extraction/decoder)",
        inputs.len(),
        configs.len(),
        decoders.len(),
        encoded_pixels as f64 / 1_000_000.0 / encode_seconds
    );
    if progressive == 0 || restarted == 0 {
        eprintln!(
            "Coverage gap: this selection has {progressive} progressive and {restarted} restart JPEGs; no missing cases were synthesized or silently counted."
        );
    }
}

#[test]
#[ignore = "explicit RS_JXL_DJXL; private Rust synthetic coefficient-frame acceptance"]
fn rust_coefficient_transcode_restores_synthetic_jpegs_with_official_decoders() {
    let decoders = decoders();
    let configs = configurations(env::var_os("RS_JXL_JPEG_MATRIX").as_deref())
        .expect("synthetic JPEG config selection");
    let inputs = fixtures::fixtures();
    assert!(
        (1..=64).contains(&inputs.len()),
        "select 1..=64 generated JPEG fixtures; no truncation"
    );
    let count = inputs.len();
    let mut names = BTreeSet::new();
    let mut checks = 0usize;
    let mut progressive = 0usize;
    let mut restarted = 0usize;
    let mut encoded_pixels = 0u64;
    let mut encoded_bytes = 0u64;
    let mut encode_seconds = 0.0;
    for (name, original) in inputs {
        assert!(
            !name.is_empty() && names.insert(name.clone()),
            "empty/duplicate fixture name: {name}"
        );
        assert!(
            !original.is_empty() && original.len() <= MAX_INPUT,
            "{name}: synthetic JPEG input byte cap"
        );
        let jpeg = extract_without_entropy(&original, Path::new(&name));
        let has_restart = jpeg.records.iter().any(|record| matches!(record, JpegRecord::Entropy(entropy) if !entropy.restart_markers.is_empty()));
        progressive += usize::from(jpeg.coding_process == Some(CodingProcess::Progressive));
        restarted += usize::from(has_restart);
        let pixels = u64::from(jpeg.width) * u64::from(jpeg.height);
        let scratch = Scratch::new();
        for options in &configs {
            let label = format!("{:?}/{:?}/{:?}", options.tier, options.order, options.cfl);
            eprintln!(
                "encoding synthetic {name} {}x{} {:?}, {} scans, restarts={has_restart}, {label}; archived entropy erased",
                jpeg.width,
                jpeg.height,
                jpeg.coding_process,
                jpeg.scan_info.len()
            );
            let start = Instant::now();
            let result = encode_coefficients(&jpeg, options);
            let seconds = start.elapsed().as_secs_f64();
            let encoded = result.unwrap_or_else(|error| {
                panic!("synthetic {name} {label}: Rust coefficient encode: {error}")
            });
            assert!(
                !encoded.is_empty() && encoded.len() <= MAX_OUTPUT,
                "{name} {label}: Rust output cap"
            );
            let bpp = encoded.len() as f64 * 8.0 / pixels as f64;
            eprintln!(
                "synthetic {name} {label}: JXL={} bytes, {bpp:.6} bpp, Rust coeffencode={seconds:.6}s, {:.3} MP/s (excludes fixture generation/extraction/decoder)",
                encoded.len(),
                pixels as f64 / 1_000_000.0 / seconds
            );
            encoded_pixels += pixels;
            encoded_bytes += encoded.len() as u64;
            encode_seconds += seconds;
            {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(scratch.path.join("encoded.jxl"))
                    .expect("create synthetic Rust JXL input");
                file.write_all(&encoded)
                    .expect("write synthetic Rust JXL input");
                file.flush().expect("flush synthetic Rust JXL input");
            }
            drop(encoded);
            for decoder in &decoders {
                let restored = decode(decoder, &scratch);
                let difference = first_difference(&original, &restored);
                assert!(
                    difference.is_none(),
                    "synthetic {name} {label} decoder={}: first mismatch={difference:?}, lengths={}/{}",
                    decoder.display(),
                    original.len(),
                    restored.len()
                );
                checks += 1;
                eprintln!(
                    "synthetic {name} {label}: exact {} JPEG bytes via {}",
                    original.len(),
                    decoder.display()
                );
            }
            scratch
                .remove("encoded.jxl")
                .expect("remove synthetic Rust JXL input");
        }
    }
    assert_eq!(checks, count * configs.len() * decoders.len());
    eprintln!(
        "{count} synthetic JPEGs, {progressive} progressive, {restarted} with restarts, {} configurations, {} decoders: {checks} byte-exact checks; all frames Rust-produced with archived entropy erased. Total JXL={encoded_bytes} bytes, Rust coeffencode={encode_seconds:.6}s, {:.3} MP/s (excludes fixture generation/extraction/decoder)",
        configs.len(),
        decoders.len(),
        encoded_pixels as f64 / 1_000_000.0 / encode_seconds
    );
}

fn check_targeted_coefficients(
    name: &str,
    original: &[u8],
    configs: &[Options],
    decoders: &[PathBuf],
) -> usize {
    assert!(!original.is_empty() && original.len() <= MAX_INPUT);
    let jpeg = extract_without_entropy(original, Path::new(name));
    let pixels = u64::from(jpeg.width) * u64::from(jpeg.height);
    let scratch = Scratch::new();
    let mut checks = 0;
    for options in configs {
        let label = format!(
            "{:?}/{:?}/{:?}/threads={}/decoding_speed={}/cooperative_ma={:?}",
            options.tier,
            options.order,
            options.cfl,
            options.threads,
            options.decoding_speed,
            options.cooperative_ma
        );
        eprintln!("targeted {name} {label}: encoding with archived entropy erased");
        let start = Instant::now();
        let result = encode_coefficients(&jpeg, options);
        let seconds = start.elapsed().as_secs_f64();
        let encoded = result.unwrap_or_else(|error| panic!("{name} {label}: {error}"));
        assert!(!encoded.is_empty() && encoded.len() <= MAX_OUTPUT);
        eprintln!(
            "targeted {name} {label}: JXL={} bytes, {:.6} bpp, Rust coeffencode={seconds:.6}s, {:.3} MP/s (excludes fixture/extraction/decoder)",
            encoded.len(),
            encoded.len() as f64 * 8.0 / pixels as f64,
            pixels as f64 / 1_000_000.0 / seconds
        );
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(scratch.path.join("encoded.jxl"))
                .expect("create targeted JXL");
            file.write_all(&encoded).expect("write targeted JXL");
            file.flush().expect("flush targeted JXL");
        }
        drop(encoded);
        for decoder in decoders {
            let restored = decode(decoder, &scratch);
            let difference = first_difference(original, &restored);
            assert!(
                difference.is_none(),
                "{name} {label} decoder={}: first mismatch={difference:?}, lengths={}/{}",
                decoder.display(),
                original.len(),
                restored.len()
            );
            checks += 1;
            eprintln!(
                "targeted {name} {label}: exact {} JPEG bytes via {}",
                original.len(),
                decoder.display()
            );
        }
        scratch.remove("encoded.jxl").unwrap();
    }
    checks
}

#[test]
#[ignore = "explicit RS_JXL_DJXL; every private source tier on two targeted JPEGs"]
fn rust_coefficient_transcode_all_tiers_with_official_decoders() {
    let decoders = decoders();
    let configs = tier_configurations();
    let inputs = fixtures::tier_fixtures();
    assert_eq!(configs.len(), 11);
    assert_eq!(inputs.len(), 2);
    let mut checks = 0;
    for (name, original) in &inputs {
        checks += check_targeted_coefficients(name, original, &configs, &decoders);
    }
    assert_eq!(checks, 2 * 11 * decoders.len());
    eprintln!(
        "all source tiers -1..=9, signed gray refinement and two-DC-group progressive/restart: {checks} byte-exact checks"
    );
}

#[test]
#[ignore = "explicit RS_JXL_DJXL; selected thread and decoding-speed combinations"]
fn rust_coefficient_transcode_execution_settings_with_official_decoders() {
    let decoders = decoders();
    let configs = execution_configurations();
    let mut checks = 0;
    for name in ["gray_seq_17x9", "rgb_seq_33x25"] {
        checks += check_targeted_coefficients(
            name,
            &fixtures::fixture(name).unwrap(),
            &configs,
            &decoders,
        );
    }
    assert_eq!(checks, 2 * 4 * decoders.len());
    eprintln!("selected threads 1/2 and decoding speeds 0/1/4: {checks} byte-exact checks");
}

#[test]
#[ignore = "explicit RS_JXL_DJXL; cooperative JPEG learners, null scope and bounded roots"]
fn rust_cooperative_jpeg_ma_with_official_decoders() {
    use crate::scheduler::WorkerLimits;
    let decoders = decoders();
    let base = configurations(None).unwrap().remove(0);
    let configs: Vec<_> = [(1, 8), (4, 1), (4, 2)]
        .into_iter()
        .map(|(threads, tasks)| Options {
            tier: SpeedTier::Tortoise,
            threads,
            cooperative_ma: Some(WorkerLimits {
                tasks,
                scratch_bytes: tasks as u64 * (32 << 20),
            }),
            ..base.clone()
        })
        .collect();
    let mut checks = 0;
    for name in ["ycbcr444_seq_513x257_rst7", "ycbcr420_seq_600x400_rst7"] {
        checks += check_targeted_coefficients(
            name,
            &fixtures::fixture(name).unwrap(),
            &configs,
            &decoders,
        );
    }
    assert_eq!(checks, 2 * 3 * decoders.len());
    eprintln!("cooperative JPEG MA, 444/420 restart fixtures: {checks} byte-exact checks");
}

#[test]
#[ignore = "explicit RS_JXL_DJXL; ICC/Exif/duplicate APP/comment/fill/tail preservation"]
fn rust_coefficient_transcode_metadata_with_official_decoders() {
    let decoders = decoders();
    let configs = configurations(None).unwrap();
    let inputs = fixtures::metadata_fixtures();
    assert_eq!(inputs.len(), 5);
    let mut checks = 0;
    for (name, original) in &inputs {
        checks += check_targeted_coefficients(name, original, &configs, &decoders);
    }
    assert_eq!(checks, 5 * 3 * decoders.len());
    eprintln!(
        "targeted ICC/Exif/unknown and duplicate APP/COM/fill/tail: {checks} byte-exact checks"
    );
}

#[test]
#[ignore = "explicit RS_JXL_DJXL; JPEG byte API Vec and file sinks, three presets/four JPEGs"]
fn rust_jpeg_bytes_adapter_with_official_decoders() {
    let decoders = decoders();
    let configs = adapter_configurations();
    let mut inputs: Vec<_> = [
        "gray_prog_refinement_24x8",
        "ycbcr420_prog_33x19_rst3",
        "rgb_seq_33x25",
    ]
    .into_iter()
    .map(|name| (name.to_owned(), fixtures::fixture(name).unwrap()))
    .collect();
    inputs.push(
        fixtures::metadata_fixtures()
            .into_iter()
            .find(|(name, _)| name == "metadata_rgb_icc_split_exif_be")
            .expect("tagged adapter fixture"),
    );
    assert_eq!(inputs.len(), 4);
    assert_eq!(configs.len(), 3);
    let mut checks = 0;
    for (name, original) in inputs {
        assert!(original.len() <= MAX_INPUT);
        let header = read_jpeg_with_limits(&original, &configs[0].limits.input.structural).unwrap();
        let pixels = u64::from(header.width) * u64::from(header.height);
        assert!((1..=MAX_PIXELS).contains(&pixels));
        drop(header);
        let scratch = Scratch::new();
        for options in &configs {
            let label = format!(
                "effort={}/threads={}/decoding_speed={}/quality={:?}",
                options.effort,
                options.threads,
                options.decoding_speed,
                options.reconstruction_quality
            );
            for file_sink in [false, true] {
                let route = if file_sink {
                    "transcode_to(file)"
                } else {
                    "transcode(Vec)"
                };
                eprintln!("byte adapter {name} {label} {route}: encoding");
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(scratch.path.join("encoded.jxl"))
                    .expect("create adapter JXL");
                let start = Instant::now();
                let (length, seconds) = if file_sink {
                    archival::transcode_to(&original, options, &mut file)
                        .unwrap_or_else(|error| panic!("{name} {label} {route}: {error}"));
                    let seconds = start.elapsed().as_secs_f64();
                    (file.metadata().unwrap().len(), seconds)
                } else {
                    let result = archival::transcode(&original, options);
                    let seconds = start.elapsed().as_secs_f64();
                    let encoded =
                        result.unwrap_or_else(|error| panic!("{name} {label} {route}: {error}"));
                    assert!(encoded.len() <= MAX_OUTPUT);
                    file.write_all(&encoded).expect("write adapter Vec JXL");
                    (encoded.len() as u64, seconds)
                };
                assert!((1..=MAX_OUTPUT as u64).contains(&length));
                file.flush().expect("flush adapter JXL");
                drop(file);
                eprintln!(
                    "byte adapter {name} {label} {route}: JXL={length} bytes, {:.6} bpp, Rust adapter={seconds:.6}s, {:.3} MP/s (includes coefficient read; file route includes sink writes; excludes oracle)",
                    length as f64 * 8.0 / pixels as f64,
                    pixels as f64 / 1_000_000.0 / seconds
                );
                for decoder in &decoders {
                    let restored = decode(decoder, &scratch);
                    let difference = first_difference(&original, &restored);
                    assert!(
                        difference.is_none(),
                        "{name} {label} {route} decoder={}: first mismatch={difference:?}, lengths={}/{}",
                        decoder.display(),
                        original.len(),
                        restored.len()
                    );
                    checks += 1;
                }
                scratch.remove("encoded.jxl").unwrap();
            }
        }
    }
    assert_eq!(checks, 4 * 3 * 2 * decoders.len());
    eprintln!(
        "JPEG byte adapter, 4 JPEGs, 3 presets, Vec/file sinks: {checks} byte-exact official decoder checks"
    );
}

#[test]
#[ignore = "explicit RS_JXL_DJXL; source JPEG search prefixes, both execution modes"]
fn rust_jpeg_search_prefixes_with_official_decoders() {
    use super::search::{SearchLimits, transcode_search};
    let decoders = decoders();
    let mut inputs: Vec<_> = [
        "gray_prog_refinement_24x8",
        "ycbcr420_prog_33x19_rst3",
        "rgb_seq_33x25",
    ]
    .into_iter()
    .map(|name| (name.to_owned(), fixtures::fixture(name).unwrap()))
    .collect();
    inputs.push(
        fixtures::metadata_fixtures()
            .into_iter()
            .find(|(name, _)| name == "metadata_rgb_icc_split_exif_be")
            .unwrap(),
    );
    let mut checks = 0;
    for (name, original) in inputs {
        let scratch = Scratch::new();
        for threads in [1, 4] {
            let mut previous = usize::MAX;
            for effort in 1..=11 {
                // Fix reconstruction quality to isolate the frame search prefix.
                let options = TranscodeOptions {
                    effort,
                    threads,
                    reconstruction_quality: Some(4),
                    ..Default::default()
                };
                let result =
                    transcode_search(&original, &options, SearchLimits::default()).unwrap();
                let expected = [(1, 3), (3, 5), (5, 7), (7, 8), (9, 9), (10, 10)]
                    .into_iter()
                    .filter(|(stage, _)| *stage <= effort.min(10))
                    .collect::<Vec<_>>();
                assert_eq!(
                    result
                        .trials
                        .iter()
                        .map(|trial| (trial.stage, trial.backend_effort))
                        .collect::<Vec<_>>(),
                    expected
                );
                assert_eq!(
                    result.data.len(),
                    result.trials.iter().map(|t| t.bytes).min().unwrap()
                );
                assert_eq!(
                    result.backend_effort,
                    result.trials[result.winner_index].backend_effort
                );
                assert!(
                    result.data.len() <= previous,
                    "{name} effort={effort}: effort prefix regressed"
                );
                previous = result.data.len();
                let path = scratch.path.join("encoded.jxl");
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .unwrap();
                file.write_all(&result.data).unwrap();
                drop(file);
                for decoder in &decoders {
                    let restored = decode(decoder, &scratch);
                    assert_eq!(
                        first_difference(&original, &restored),
                        None,
                        "{name} effort={effort} threads={threads}, {}",
                        decoder.display()
                    );
                    checks += 1;
                }
                scratch.remove("encoded.jxl").unwrap();
            }
        }
    }
    assert_eq!(checks, 4 * 11 * 2 * decoders.len());
    eprintln!("{checks} JPEG search-prefix original-byte checks passed");
}

#[test]
fn private_composer_rejects_invalid_options_and_exhausted_budgets() {
    let original = fixtures::fixture("rgb_seq_8x8").unwrap();
    let jpeg = extract_without_entropy(&original, Path::new("budget_rgb_8x8"));
    let base = configurations(None).unwrap().remove(0);
    let cases = [
        Options {
            threads: 0,
            ..base.clone()
        },
        Options {
            threads: 257,
            ..base.clone()
        },
        Options {
            decoding_speed: 5,
            ..base.clone()
        },
        Options {
            reconstruction_quality: 12,
            ..base.clone()
        },
        Options {
            sample_limit: 0,
            ..base.clone()
        },
        Options {
            max_output_bytes: 0,
            ..base.clone()
        },
        Options {
            max_coefficients: 0,
            ..base.clone()
        },
        Options {
            max_tokens: 0,
            ..base.clone()
        },
        Options {
            max_output_bytes: 1,
            ..base.clone()
        },
        Options {
            max_coefficients: 3 * 64 - 1,
            ..base.clone()
        },
        Options {
            max_tokens: 1,
            ..base.clone()
        },
    ];
    let components = jpeg.components.clone();
    let records = jpeg.records.clone();
    for options in cases {
        let Err(error) = encode_coefficients(&jpeg, &options) else {
            panic!("invalid option/budget must fail without output");
        };
        assert!(!error.to_string().is_empty());
        assert_eq!(jpeg.components, components);
        assert_eq!(jpeg.records, records);
    }
    let exact_coefficients = Options {
        max_coefficients: 3 * 64,
        ..base
    };
    let encoded = encode_coefficients(&jpeg, &exact_coefficients)
        .expect("exact coefficient cap permits valid tiny input after errors");
    assert!(!encoded.is_empty() && encoded.len() <= MAX_OUTPUT);
}

#[test]
fn private_composer_rejects_mutated_input_and_strictly_invalid_icc() {
    let original = fixtures::fixture("rgb_seq_8x8").unwrap();
    let jpeg = extract_without_entropy(&original, Path::new("mutated_input"));
    let options = configurations(None).unwrap().remove(0);
    let mut width = jpeg.clone();
    width.width = 0;
    let mut coefficients = jpeg.clone();
    coefficients.components[0].coefficients.pop();
    let mut app = jpeg.clone();
    app.app_data[0].pop();
    let mut scan = jpeg;
    scan.scan_info[0].mcu_cols += 1;
    for changed in [width, coefficients, app, scan] {
        assert!(encode_coefficients(&changed, &options).is_err());
    }
    let cases = fixtures::malformed_metadata_fixtures();
    assert_eq!(cases.len(), 6);
    for (name, original) in cases {
        let jpeg = extract_without_entropy(&original, Path::new(&name));
        let Err(error) = encode_coefficients(&jpeg, &options) else {
            panic!("{name}: invalid ICC accepted");
        };
        assert!(
            matches!(error, crate::Error::InvalidInput(_)),
            "{name}: {error}"
        );
    }
}

#[test]
fn byte_adapter_encoding_errors_leave_file_sink_untouched() {
    let original = fixtures::fixture("rgb_seq_8x8").unwrap();
    let base = adapter_configurations().remove(0);
    let mut options = vec![base.clone(); 8];
    options[0].effort = 0;
    options[1].limits.max_output_bytes = 0;
    options[2].limits.max_tokens = 1;
    options[3].limits.max_coefficients = 1;
    options[4].limits.sample_limit = 0;
    options[5].limits.input.structural.max_input_bytes = original.len() - 1;
    options[6].limits.input.max_coefficient_bytes = 1;
    options[7].limits.input.max_block_visits = 0;
    let scratch = Scratch::new();
    let path = scratch.path.join("encoded.jxl");
    let sentinel = b"existing sink contents";
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(sentinel).unwrap();
        for settings in options {
            assert!(archival::transcode_to(&original, &settings, &mut file).is_err());
            assert_eq!(file.metadata().unwrap().len(), sentinel.len() as u64);
        }
        for malformed in [b"not JPEG".as_slice(), &original[..original.len() - 1]] {
            assert!(archival::transcode_to(malformed, &base, &mut file).is_err());
        }
        for (name, malformed) in fixtures::malformed_metadata_fixtures() {
            assert!(
                archival::transcode_to(&malformed, &base, &mut file).is_err(),
                "{name}"
            );
        }
        file.flush().unwrap();
    }
    assert_eq!(read_bounded(&path, MAX_INPUT).unwrap(), sentinel);
}

#[test]
fn byte_adapter_propagates_partial_sink_io_failure() {
    struct FailingSink(Vec<u8>);
    impl Write for FailingSink {
        fn write(&mut self, input: &[u8]) -> io::Result<usize> {
            if self.0.len() == 7 {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "test sink failure",
                ));
            }
            let length = input.len().min(3).min(7 - self.0.len());
            self.0.extend_from_slice(&input[..length]);
            Ok(length)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let original = fixtures::fixture("gray_seq_1x1").unwrap();
    let mut sink = FailingSink(Vec::new());
    let error = archival::transcode_to(&original, &adapter_configurations().remove(0), &mut sink)
        .unwrap_err();
    assert!(
        matches!(error, crate::Error::Io(ref io) if io.kind() == io::ErrorKind::PermissionDenied)
    );
    assert_eq!(sink.0.len(), 7);
}

#[test]
fn targeted_schedules_cover_all_tiers_and_requested_execution_settings() {
    let tiers = tier_configurations();
    assert_eq!(
        tiers.iter().map(|o| o.tier as i8).collect::<Vec<_>>(),
        (-1i8..=9).collect::<Vec<_>>()
    );
    assert!(
        tiers
            .iter()
            .all(|o| o.threads == 1 && o.decoding_speed == 0)
    );
    let settings = execution_configurations();
    assert_eq!(settings.len(), 4);
    assert_eq!(
        settings.iter().map(|o| o.threads).collect::<BTreeSet<_>>(),
        BTreeSet::from([1, 2])
    );
    assert_eq!(
        settings
            .iter()
            .map(|o| o.decoding_speed)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1, 4])
    );
    assert_eq!(adapter_configurations().len(), 3);
}

#[test]
fn oracle_configurations_are_bounded_and_cover_requested_modes() {
    let defaults = configurations(None).unwrap();
    assert_eq!(defaults.len(), 3);
    assert_eq!(defaults[0].order, Options::default().order);
    assert_eq!(
        (defaults[1].tier, defaults[1].order, defaults[1].cfl),
        (SpeedTier::Falcon, OrderPolicy::Default, CflMode::Disabled)
    );
    assert_eq!(
        (defaults[2].tier, defaults[2].order, defaults[2].cfl),
        (SpeedTier::Tortoise, OrderPolicy::Adaptive, CflMode::Enabled)
    );
    assert_eq!(configurations(Some(OsStr::new("0"))).unwrap().len(), 3);
    let matrix = configurations(Some(OsStr::new("1"))).unwrap();
    assert_eq!(matrix.len(), 12);
    for (index, config) in matrix.iter().enumerate() {
        assert_eq!(config.max_output_bytes, MAX_OUTPUT);
        assert_eq!(
            config.max_coefficients * std::mem::size_of::<i32>(),
            MAX_COEFFICIENT_BYTES
        );
        assert!(!matrix[..index].iter().any(|old| old.tier == config.tier
            && old.order == config.order
            && old.cfl == config.cfl));
    }
    for invalid in ["", "yes", "2", "all"] {
        assert!(configurations(Some(OsStr::new(invalid))).is_err());
    }
}

#[test]
fn byte_comparison_rejects_changes_truncation_and_trailing_bytes() {
    assert_eq!(first_difference(b"jpeg", b"jpeg"), None);
    assert_eq!(first_difference(b"jpeg", b"jXeg"), Some(1));
    assert_eq!(first_difference(b"jpeg", b"jpe"), Some(3));
    assert_eq!(first_difference(b"jpeg", b"jpeg!"), Some(4));
    assert_eq!(first_difference(b"", b""), None);
}

#[test]
fn bounded_files_reject_oversize_and_cleanup_refuses_unowned_names() {
    let scratch = Scratch::new();
    let path = scratch.path.join("encoded.jxl");
    fs::write(&path, [1, 2, 3]).unwrap();
    assert_eq!(read_bounded(&path, 3).unwrap(), [1, 2, 3]);
    assert!(read_bounded(&path, 2).is_err());
    assert!(check_size(&path, 3).unwrap());
    assert!(check_size(&path, 2).is_err());
    assert!(scratch.remove("../encoded.jxl").is_err());
    scratch.remove("encoded.jxl").unwrap();
    assert!(!check_size(&path, 3).unwrap());
}
