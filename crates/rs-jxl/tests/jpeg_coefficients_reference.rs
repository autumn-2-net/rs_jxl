//! Independent, development-only libjpeg-turbo coefficient comparison.
//! RS_JXL_JPEG_COEFFICIENT_ORACLE names an executable, not a codec dependency.
//! RS_JXL_JPEG_INPUTS is an OS path list of JPEG files / directories. Directory
//! discovery is nonrecursive; the first 17 unique, sorted canonical JPEG paths
//! are selected, with any count truncation reported. Resource failures fail.
//!
//! Oracle argv: input.jpg output.bin. Binary output: JCO1, little-endian u32
//! width/height/component_count, then for EACH component u32 id/h/v/block_width/
//! block_height followed by block-major, natural-order little-endian i16 data.
//! libjpeg reports unpadded block dimensions and discards dummy MCU blocks.
//! No JPEG writer, JXL encoding, IDCT or pixel comparison is exercised here.
//!
//! Parent runs centrally (this test never builds the external oracle):
//! cargo test -p rs-jxl --test jpeg_coefficients_reference -- --ignored --nocapture

use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::jpeg::coefficient_reader::{CoefficientReadLimits, read_coefficients_with_limits};
use rs_jxl::jpeg::data::{BLOCK_SIZE, CodingProcess, JpegData, MAX_COMPONENTS};
use rs_jxl::jpeg::reader::{ReadLimits, read_jpeg_with_limits};

const MAX_FIXTURES: usize = 17;
const MAX_DISCOVERY_ENTRIES: usize = 4096;
const MAX_INPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_COEFFICIENT_BYTES: usize = 256 * 1024 * 1024;
const MAX_OUTPUT_BYTES: u64 = 256 * 1024 * 1024;
const MAX_PIXELS: u64 = 64_000_000;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
const TEMP_PREFIX: &str = "rs-jxl-jpeg-coefficients-";
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn is_link(metadata: &Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0 // Reparse points, including junctions.
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn is_jpeg(path: &Path) -> bool {
    path.extension().is_some_and(|extension| {
        extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg")
    })
}

fn discover(inputs: &std::ffi::OsStr) -> Vec<PathBuf> {
    let mut files = BTreeSet::new();
    let mut inspected = 0usize;
    let mut outside_scope = 0usize;
    let mut count_entry = || {
        inspected += 1;
        assert!(
            inspected <= MAX_DISCOVERY_ENTRIES,
            "fixture discovery exceeds {MAX_DISCOVERY_ENTRIES} entries; narrow RS_JXL_JPEG_INPUTS"
        );
    };
    for path in env::split_paths(inputs) {
        count_entry();
        assert!(
            !path.as_os_str().is_empty(),
            "empty RS_JXL_JPEG_INPUTS entry"
        );
        let metadata = fs::symlink_metadata(&path)
            .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
        assert!(
            !is_link(&metadata),
            "linked fixture root is not supported: {}",
            path.display()
        );
        if metadata.is_dir() {
            for entry in fs::read_dir(&path).expect("read fixture directory") {
                count_entry();
                let entry = entry.expect("read fixture directory entry");
                let path = entry.path();
                let metadata = fs::symlink_metadata(&path).expect("inspect fixture entry");
                if metadata.is_dir() || !is_jpeg(&path) {
                    outside_scope += 1;
                    continue; // Explicitly nonrecursive; these are not selected JPEGs.
                }
                assert!(
                    !is_link(&metadata) && metadata.is_file(),
                    "JPEG candidate must be a regular, unlinked file: {}",
                    path.display()
                );
                files.insert(path.canonicalize().expect("resolve JPEG fixture"));
            }
        } else {
            assert!(
                metadata.is_file() && is_jpeg(&path),
                "fixture must be a JPEG file or directory: {}",
                path.display()
            );
            files.insert(path.canonicalize().expect("resolve JPEG fixture"));
        }
    }
    assert!(
        !files.is_empty(),
        "RS_JXL_JPEG_INPUTS selected no JPEG fixtures"
    );
    let selected = files.len().min(MAX_FIXTURES);
    eprintln!(
        "JPEG oracle selection: {selected}/{} unique JPEGs; {} beyond first-{MAX_FIXTURES} cap; \
        {outside_scope} non-JPEG entries/subdirectories excluded (nonrecursive)",
        files.len(),
        files.len() - selected
    );
    files.into_iter().take(MAX_FIXTURES).collect()
}

fn read_input(path: &Path) -> Vec<u8> {
    let mut file = File::open(path).expect("open JPEG fixture");
    let metadata = file.metadata().expect("inspect open JPEG fixture");
    assert!(
        metadata.is_file() && metadata.len() <= MAX_INPUT_BYTES as u64,
        "{}: input is not a regular file or exceeds 16 MiB ({} bytes)",
        path.display(),
        metadata.len()
    );
    let length = usize::try_from(metadata.len()).expect("bounded input length");
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .expect("reserve bounded JPEG input");
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)
        .expect("read bounded JPEG fixture");
    let mut extra = [0];
    assert_eq!(
        file.read(&mut extra).expect("check fixture EOF"),
        0,
        "{}: input grew during read",
        path.display()
    );
    bytes
}

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
            let path = root.join(format!("{TEMP_PREFIX}{}-{stamp}-{id}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self { root, path },
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        }
        panic!("cannot create unique JPEG oracle scratch directory");
    }

    fn cleanup(&self) -> io::Result<()> {
        let metadata = fs::symlink_metadata(&self.path)?;
        let named = self
            .path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with(TEMP_PREFIX));
        if !named
            || is_link(&metadata)
            || !metadata.is_dir()
            || self.path.parent() != Some(self.root.as_path())
            || self.path.canonicalize()? != self.path
            || self.root.canonicalize()? != self.root
        {
            return Err(io::Error::other(
                "refusing cleanup of changed/out-of-root scratch path",
            ));
        }
        // Only remove our two known files, then the empty directory. Never
        // recursively delete an oracle-created tree or follow a replacement link.
        for name in ["input.jpg", "output.bin"] {
            let path = self.path.join(name);
            let metadata = match fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            if is_link(&metadata) || !metadata.is_file() || path.canonicalize()? != path {
                return Err(io::Error::other(
                    "refusing cleanup of replaced scratch file",
                ));
            }
            fs::remove_file(&path)?;
        }
        fs::remove_dir(&self.path)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Err(error) = self.cleanup() {
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
        if !matches!(self.0.try_wait(), Ok(Some(_)))
            && let Err(error) = self.0.kill()
        {
            eprintln!("kill JPEG coefficient oracle: {error}");
        }
        if let Err(error) = self.0.wait() {
            if thread::panicking() {
                eprintln!("reap JPEG coefficient oracle: {error}");
            } else {
                panic!("reap JPEG coefficient oracle: {error}");
            }
        }
    }
}

fn check_output_size(path: &Path) -> Option<u64> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return None,
        Err(error) => panic!("inspect oracle output: {error}"),
    };
    assert!(
        metadata.is_file() && !is_link(&metadata),
        "oracle output is not a regular, unlinked file"
    );
    assert!(
        metadata.len() <= MAX_OUTPUT_BYTES,
        "oracle output exceeds 256 MiB: {} bytes",
        metadata.len()
    );
    Some(metadata.len())
}

fn run_oracle(executable: &Path, scratch: &ScratchDir, context: &Path) -> PathBuf {
    let output = scratch.path.join("output.bin");
    let mut command = Command::new(executable);
    command
        .arg(scratch.path.join("input.jpg"))
        .arg(&output)
        .current_dir(&scratch.path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW.
    }
    let mut child = ReapOnDrop(command.spawn().unwrap_or_else(|error| {
        panic!(
            "{}: spawn {}: {error}",
            context.display(),
            executable.display()
        )
    }));
    let start = Instant::now();
    loop {
        // A failed cap check unwinds through ReapOnDrop before scratch cleanup.
        check_output_size(&output);
        if let Some(status) = child.0.try_wait().expect("poll coefficient oracle") {
            assert!(
                status.success(),
                "{}: coefficient oracle exited {status}",
                context.display()
            );
            assert!(
                check_output_size(&output).is_some(),
                "oracle produced no output file"
            );
            return output;
        }
        assert!(
            start.elapsed() < COMMAND_TIMEOUT,
            "{}: coefficient oracle exceeded {COMMAND_TIMEOUT:?}; kill/reap on unwind",
            context.display()
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn word(reader: &mut impl Read) -> Result<u32, String> {
    let mut bytes = [0; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("truncated JCO1 header: {error}"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn compare_coefficients(reader: &mut impl Read, jpeg: &JpegData) -> Result<u64, String> {
    let mut magic = [0; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|error| format!("truncated JCO1 magic: {error}"))?;
    if &magic != b"JCO1" {
        return Err("invalid JCO1 magic".into());
    }
    let (width, height, count) = (word(reader)?, word(reader)?, word(reader)?);
    if width == 0
        || height == 0
        || width != jpeg.width
        || height != jpeg.height
        || count == 0
        || count > MAX_COMPONENTS as u32
        || count as usize != jpeg.components.len()
    {
        return Err(format!(
            "JCO1 frame mismatch: {width}x{height}, {count} components"
        ));
    }
    let max_h = jpeg
        .components
        .iter()
        .map(|c| c.h_samp_factor)
        .max()
        .unwrap();
    let max_v = jpeg
        .components
        .iter()
        .map(|c| c.v_samp_factor)
        .max()
        .unwrap();
    if max_h == 0 || max_v == 0 {
        return Err("invalid Rust sampling geometry".into());
    }
    let mut output_bytes = 16u64;
    let mut compared = 0u64;
    for (index, component) in jpeg.components.iter().enumerate() {
        let (id, h, v, cols, rows) = (
            word(reader)?,
            word(reader)?,
            word(reader)?,
            word(reader)?,
            word(reader)?,
        );
        if id != component.id
            || h != component.h_samp_factor
            || v != component.v_samp_factor
            || h == 0
            || v == 0
        {
            return Err(format!(
                "JCO1 component {index}: ID/sampling mismatch ({id}, {h}, {v})"
            ));
        }
        let actual_cols = (u64::from(width) * u64::from(h)).div_ceil(8 * u64::from(max_h));
        let actual_rows = (u64::from(height) * u64::from(v)).div_ceil(8 * u64::from(max_v));
        if u64::from(cols) != actual_cols || u64::from(rows) != actual_rows {
            return Err(format!(
                "JCO1 component {index}: unpadded dimensions {cols}x{rows}, expected {actual_cols}x{actual_rows}"
            ));
        }
        let stored = u64::from(component.width_in_blocks)
            .checked_mul(u64::from(component.height_in_blocks))
            .and_then(|n| n.checked_mul(BLOCK_SIZE as u64))
            .ok_or("Rust storage dimensions overflow")?;
        if component.width_in_blocks < cols
            || component.height_in_blocks < rows
            || stored != component.coefficients.len() as u64
        {
            return Err(format!("component {index}: invalid Rust padded storage"));
        }
        let values = actual_cols
            .checked_mul(actual_rows)
            .and_then(|n| n.checked_mul(BLOCK_SIZE as u64))
            .ok_or("JCO1 dimensions overflow")?;
        output_bytes = values
            .checked_mul(2)
            .and_then(|n| n.checked_add(20))
            .and_then(|n| output_bytes.checked_add(n))
            .ok_or("JCO1 length overflow")?;
        if output_bytes > MAX_OUTPUT_BYTES {
            return Err("JCO1 payload exceeds 256 MiB".into());
        }
        for y in 0..rows {
            for x in 0..cols {
                let mut block = [0u8; BLOCK_SIZE * 2];
                reader.read_exact(&mut block).map_err(|error| {
                    format!(
                        "JCO1 component {index}, block ({x},{y}): truncated coefficients: {error}"
                    )
                })?;
                let base =
                    (y as usize * component.width_in_blocks as usize + x as usize) * BLOCK_SIZE;
                for (k, bytes) in block.chunks_exact(2).enumerate() {
                    let oracle = i16::from_le_bytes([bytes[0], bytes[1]]);
                    let rust = component.coefficients[base + k];
                    if oracle != rust {
                        return Err(format!(
                            "component {index} (id {id}), block ({x},{y}), natural coefficient {k}: Rust {rust}, oracle {oracle}"
                        ));
                    }
                }
            }
        }
        compared += values;
    }
    let mut extra = [0];
    match reader.read_exact(&mut extra) {
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(compared),
        Ok(()) => Err("trailing bytes after JCO1 coefficients".into()),
        Err(error) => Err(format!("JCO1 EOF check: {error}")),
    }
}

#[test]
#[ignore = "requires RS_JXL_JPEG_COEFFICIENT_ORACLE executable and RS_JXL_JPEG_INPUTS path list"]
fn real_jpeg_coefficients_match_independent_libjpeg_turbo() {
    let oracle = env::var_os("RS_JXL_JPEG_COEFFICIENT_ORACLE").expect(
        "RS_JXL_JPEG_COEFFICIENT_ORACLE is required (external development-only executable)",
    );
    assert!(!oracle.is_empty(), "empty RS_JXL_JPEG_COEFFICIENT_ORACLE");
    let oracle = PathBuf::from(oracle)
        .canonicalize()
        .expect("resolve coefficient oracle executable");
    assert!(
        oracle.is_file(),
        "coefficient oracle is not a file: {}",
        oracle.display()
    );
    let inputs = env::var_os("RS_JXL_JPEG_INPUTS").expect("RS_JXL_JPEG_INPUTS is required");
    let fixtures = discover(&inputs);
    let limits = CoefficientReadLimits {
        structural: ReadLimits {
            max_input_bytes: MAX_INPUT_BYTES,
            ..Default::default()
        },
        max_coefficient_bytes: MAX_COEFFICIENT_BYTES,
        ..Default::default()
    };
    let (mut progressive, mut restarted, mut total_coefficients) = (0usize, 0usize, 0u64);
    for path in &fixtures {
        let input = read_input(path);
        // Check the external oracle's pixel ceiling before allocating coefficients.
        let header = read_jpeg_with_limits(&input, &limits.structural)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        assert!(
            u64::from(header.width) * u64::from(header.height) <= MAX_PIXELS,
            "{}: exceeds oracle's 64,000,000-pixel limit",
            path.display()
        );
        drop(header);
        let jpeg = read_coefficients_with_limits(&input, &limits)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        assert!(
            jpeg.reconstruct().expect("archive reconstruction") == input,
            "{}: original JPEG archive bytes differ",
            path.display()
        );
        let scratch = ScratchDir::new();
        {
            let mut snapshot = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(scratch.path.join("input.jpg"))
                .expect("create private input snapshot");
            snapshot
                .write_all(&input)
                .expect("write bounded input snapshot");
            snapshot.flush().expect("flush input snapshot");
        }
        let output = run_oracle(&oracle, &scratch, path);
        let file = File::open(output).expect("open bounded oracle output");
        let mut reader = BufReader::new(file.take(MAX_OUTPUT_BYTES + 1));
        let compared = compare_coefficients(&mut reader, &jpeg)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        drop(reader); // Windows cleanup requires all output handles to be closed.
        progressive += usize::from(jpeg.coding_process == Some(CodingProcess::Progressive));
        let has_restart = jpeg.records.iter().any(|record| {
            matches!(record,
            rs_jxl::jpeg::data::JpegRecord::Entropy(data) if !data.restart_markers.is_empty())
        });
        restarted += usize::from(has_restart);
        total_coefficients += compared;
        eprintln!(
            "{}: {}x{}, {:?}, {} scans, restarts={has_restart}, {compared} real coefficients bit-exact; archive exact",
            path.display(),
            jpeg.width,
            jpeg.height,
            jpeg.coding_process,
            jpeg.scan_info.len()
        );
    }
    eprintln!(
        "{} JPEGs, {progressive} progressive, {restarted} with restart markers; \
        {total_coefficients} real coefficients matched independent libjpeg-turbo; NOT a JPEG-to-JXL test",
        fixtures.len()
    );
    if restarted == 0 {
        eprintln!(
            "Coverage gap: selected fixtures contain no restart markers; supply a real restart JPEG in RS_JXL_JPEG_INPUTS."
        );
    }
}

fn protocol_fixture() -> (JpegData, Vec<u8>) {
    use rs_jxl::jpeg::data::Component;
    let mut jpeg = JpegData {
        width: 17,
        height: 9,
        ..Default::default()
    };
    let mut bytes = b"JCO1".to_vec();
    for value in [17u32, 9, 3] {
        bytes.extend(value.to_le_bytes());
    }
    for (id, h, v, cols, rows, padded_cols) in
        [(1, 2, 2, 3, 2, 4), (2, 1, 1, 2, 1, 2), (3, 1, 1, 2, 1, 2)]
    {
        let mut component = Component {
            id,
            h_samp_factor: h,
            v_samp_factor: v,
            width_in_blocks: padded_cols,
            height_in_blocks: rows,
            coefficients: vec![12345; padded_cols as usize * rows as usize * BLOCK_SIZE],
            ..Default::default()
        };
        for value in [id, h, v, cols, rows] {
            bytes.extend(value.to_le_bytes());
        }
        for y in 0..rows {
            for x in 0..cols {
                for k in 0..BLOCK_SIZE {
                    let value = (id as i16 * 100 + y as i16 * 20 + x as i16 * 3 + k as i16) - 180;
                    component.coefficients
                        [(y as usize * padded_cols as usize + x as usize) * BLOCK_SIZE + k] = value;
                    bytes.extend(value.to_le_bytes());
                }
            }
        }
        jpeg.components.push(component);
    }
    (jpeg, bytes)
}

#[test]
fn jco1_parser_compares_real_blocks_with_padded_rust_stride() {
    let (jpeg, bytes) = protocol_fixture();
    assert_eq!(
        compare_coefficients(&mut bytes.as_slice(), &jpeg),
        Ok(10 * 64)
    );
    let mut mismatch = bytes.clone();
    mismatch[36] ^= 1;
    assert!(
        compare_coefficients(&mut mismatch.as_slice(), &jpeg)
            .unwrap_err()
            .contains("natural coefficient 0")
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(
        compare_coefficients(&mut trailing.as_slice(), &jpeg)
            .unwrap_err()
            .contains("trailing bytes")
    );
    for length in 0..bytes.len() {
        assert!(
            compare_coefficients(&mut &bytes[..length], &jpeg).is_err(),
            "prefix {length}"
        );
    }
}

#[test]
fn jco1_parser_rejects_each_header_field_and_oversized_dimensions() {
    let (jpeg, bytes) = protocol_fixture();
    for offset in [
        0usize, 4, 8, 12, 16, 20, 24, 28, 32, 804, 808, 812, 816, 820, 1080, 1084, 1088, 1092, 1096,
    ] {
        let mut invalid = bytes.clone();
        invalid[offset..offset + 4].fill(0xff);
        assert!(
            compare_coefficients(&mut invalid.as_slice(), &jpeg).is_err(),
            "header byte {offset}"
        );
    }
}
