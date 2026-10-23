//! Independent jbrd compatibility test, NOT a Rust JPEG-to-JXL encoder test.
//! The official encoder supplies only the test coefficient frame. Rust replaces
//! its reconstruction box; official decoders must then reproduce the JPEG bytes.

use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use rs_jxl::jpeg::{
    coefficient_reader::{CoefficientReadLimits, read_coefficients_with_limits},
    data::JpegRecord,
    reconstruction_brotli::compress_parts,
    reconstruction_metadata::serialize_reconstruction_metadata,
};

const SIGNATURE: &[u8] = b"\0\0\0\x0cJXL \r\n\x87\n";
const MAX_INPUT: usize = 16 * 1024 * 1024;
const MAX_OUTPUT: usize = 64 * 1024 * 1024;
const FILES: [&str; 4] = ["original.jxl", "patched.jxl", "decoded.jpg", "stderr.txt"];

fn read_bounded(path: &Path, maximum: usize) -> Vec<u8> {
    let file = fs::File::open(path).unwrap();
    assert!(file.metadata().unwrap().len() <= maximum as u64);
    let mut data = Vec::new();
    file.take(maximum as u64 + 1)
        .read_to_end(&mut data)
        .unwrap();
    assert!(data.len() <= maximum);
    data
}

fn inputs() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let roots = env::var_os("RS_JXL_JPEG_INPUTS").expect("explicit JPEG input list required");
    for root in env::split_paths(&roots) {
        if root.is_file() {
            paths.push(root.canonicalize().unwrap());
        } else {
            for (index, entry) in fs::read_dir(root).unwrap().enumerate() {
                assert!(index < 4096, "limit nonrecursive JPEG discovery");
                let entry = entry.unwrap();
                let path = entry.path();
                if entry.file_type().unwrap().is_file()
                    && path.extension().is_some_and(|e| {
                        e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg")
                    })
                {
                    paths.push(path.canonicalize().unwrap());
                }
            }
        }
    }
    paths.sort();
    paths.dedup();
    assert!(
        !paths.is_empty() && paths.len() <= 17,
        "select 1..=17 JPEGs"
    );
    paths
}

fn visit_boxes<'a>(
    data: &'a [u8],
    mut visit: impl FnMut(&[u8], &'a [u8], &'a [u8]) -> Result<(), &'static str>,
) -> Result<(), &'static str> {
    if !data.starts_with(SIGNATURE) || data.len() > MAX_OUTPUT {
        return Err("not a bounded JXL container");
    }
    let mut offset = 0usize;
    while offset < data.len() {
        let remaining = &data[offset..];
        if remaining.len() < 8 {
            return Err("truncated box header");
        }
        let short = u32::from_be_bytes(remaining[..4].try_into().unwrap());
        let (size, header) = match short {
            0 => (remaining.len(), 8),
            1 => {
                if remaining.len() < 16 {
                    return Err("truncated extended box");
                }
                (
                    usize::try_from(u64::from_be_bytes(remaining[8..16].try_into().unwrap()))
                        .map_err(|_| "large box")?,
                    16,
                )
            }
            size => (size as usize, 8),
        };
        if size < header || size > remaining.len() {
            return Err("invalid box size");
        }
        visit(
            &remaining[4..8],
            &remaining[header..size],
            &remaining[..size],
        )?;
        offset += size;
    }
    Ok(())
}

fn replace_jbrd(data: &[u8], payload: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut output = Vec::new();
    let mut count = 0;
    visit_boxes(data, |kind, _, original| {
        let encoded = if kind == b"jbrd" {
            count += 1;
            let length = payload.len().checked_add(8).ok_or("box overflow")?;
            let length = u32::try_from(length).map_err(|_| "box overflow")?;
            if output
                .len()
                .checked_add(length as usize)
                .is_none_or(|n| n > MAX_OUTPUT)
            {
                return Err("output limit");
            }
            output.extend_from_slice(&length.to_be_bytes());
            output.extend_from_slice(b"jbrd");
            payload
        } else {
            original
        };
        if output
            .len()
            .checked_add(encoded.len())
            .is_none_or(|n| n > MAX_OUTPUT)
        {
            return Err("output limit");
        }
        output.extend_from_slice(encoded);
        Ok(())
    })?;
    if count != 1 {
        return Err("expected exactly one reconstruction box");
    }
    Ok(output)
}

fn fixture_codestream(data: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut code = Vec::new();
    let mut next = 0u32;
    let mut complete = false;
    visit_boxes(data, |kind, payload, _| {
        let append = match kind {
            b"jxlc" => {
                if complete || next != 0 {
                    return Err("mixed/duplicate codestream boxes");
                }
                complete = true;
                payload
            }
            b"jxlp" => {
                if complete || payload.len() < 4 {
                    return Err("invalid partial codestream");
                }
                let index = u32::from_be_bytes(payload[..4].try_into().unwrap());
                if index & 0x7fff_ffff != next {
                    return Err("partial codestream sequence");
                }
                complete = index & 0x8000_0000 != 0;
                next += 1;
                &payload[4..]
            }
            _ => return Ok(()),
        };
        if code
            .len()
            .checked_add(append.len())
            .is_none_or(|n| n > MAX_OUTPUT)
        {
            return Err("codestream limit");
        }
        code.extend_from_slice(append);
        Ok(())
    })?;
    if !complete || !code.starts_with(&[0xff, 0x0a]) {
        return Err("missing/unfinished bare codestream");
    }
    Ok(code)
}

fn is_link(metadata: &fs::Metadata) -> bool {
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

struct Scratch {
    root: PathBuf,
    path: PathBuf,
}
impl Scratch {
    fn new() -> Self {
        let root = env::temp_dir().canonicalize().unwrap();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = root.join(format!("rs-jxl-jbrd-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self { root, path }
    }
    fn cleanup(&self) -> std::io::Result<()> {
        if self.path.parent() != Some(&self.root)
            || self.path.canonicalize()? != self.path
            || is_link(&fs::symlink_metadata(&self.path)?)
            || self.root.canonicalize()? != self.root
        {
            return Err(std::io::Error::other("changed private scratch path"));
        }
        // No recursive deletion: only remove the four known files we created.
        for name in FILES {
            let path = self.path.join(name);
            let metadata = match fs::symlink_metadata(&path) {
                Ok(m) => m,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            };
            if !metadata.is_file() || is_link(&metadata) || path.canonicalize()? != path {
                return Err(std::io::Error::other("changed private scratch file"));
            }
            fs::remove_file(path)?;
        }
        fs::remove_dir(&self.path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!("scratch cleanup: {e}");
        }
    }
}

struct Reap(Child);
impl Drop for Reap {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

fn run(command: &mut Command, scratch: &Scratch, output: &Path) {
    let log = scratch.path.join("stderr.txt");
    let mut child = Reap(
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(fs::File::create(&log).unwrap())
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    let status = loop {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "oracle timeout: {command:?}"
        );
        assert!(
            fs::metadata(&log).unwrap().len() <= 65536,
            "oracle log limit"
        );
        if let Ok(meta) = fs::metadata(output) {
            assert!(meta.len() <= MAX_OUTPUT as u64);
        }
        if let Some(status) = child.0.try_wait().unwrap() {
            break status;
        }
        thread::sleep(Duration::from_millis(20));
    };
    assert!(
        status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&read_bounded(&log, 65536))
    );
}

#[test]
#[ignore = "explicit RS_JXL_CJXL/RS_JXL_DJXL/RS_JXL_JPEG_INPUTS; metadata-only independent oracle"]
fn rust_reconstruction_box_restores_original_jpeg_with_official_decoders() {
    let encoder = env::var_os("RS_JXL_CJXL").expect("official test encoder required");
    let decoders: Vec<_> =
        env::split_paths(&env::var_os("RS_JXL_DJXL").expect("official decoders required"))
            .collect();
    assert!(!decoders.is_empty());
    let scratch = Scratch::new();
    let mut checks = 0;
    for path in inputs() {
        let original = read_bounded(&path, MAX_INPUT);
        let mut jpeg =
            read_coefficients_with_limits(&original, &CoefficientReadLimits::default()).unwrap();
        for record in &mut jpeg.records {
            if let JpegRecord::Entropy(entropy) = record {
                entropy.bytes.clear();
                entropy.bytes.shrink_to_fit();
            }
        }
        let parts = serialize_reconstruction_metadata(&jpeg, MAX_OUTPUT).unwrap();
        let official = scratch.path.join("original.jxl");
        run(
            Command::new(&encoder).arg(&path).arg(&official).args([
                "-d",
                "0",
                "-e",
                "3",
                "--lossless_jpeg=1",
                "--num_threads=1",
            ]),
            &scratch,
            &official,
        );
        let source = read_bounded(&official, MAX_OUTPUT);
        let codestream = fixture_codestream(&source).unwrap();
        for quality in [3, 9] {
            let jbrd = compress_parts(&parts.header, &parts.payload, quality, MAX_OUTPUT).unwrap();
            let patched = scratch.path.join("patched.jxl");
            for (container, bytes) in [
                ("official", replace_jbrd(&source, &jbrd).unwrap()),
                (
                    "Rust",
                    rs_jxl::jpeg::container::wrap(&codestream, &jbrd, 5, MAX_OUTPUT).unwrap(),
                ),
            ] {
                fs::write(&patched, bytes).unwrap();
                for decoder in &decoders {
                    let decoded = scratch.path.join("decoded.jpg");
                    run(
                        Command::new(decoder)
                            .arg(&patched)
                            .arg(&decoded)
                            .args(["--num_threads=1", "--quiet"]),
                        &scratch,
                        &decoded,
                    );
                    let restored = read_bounded(&decoded, MAX_INPUT);
                    let difference = original.iter().zip(&restored).position(|(a, b)| a != b);
                    assert!(
                        difference.is_none() && original.len() == restored.len(),
                        "{} quality={quality} container={container} decoder={} first difference={difference:?}, lengths={}/{}",
                        path.display(),
                        decoder.display(),
                        original.len(),
                        restored.len()
                    );
                    checks += 1;
                    fs::remove_file(decoded).unwrap();
                }
            }
            eprintln!(
                "{}: Rust jbrd={} bytes, Brotli={quality}, official/Rust containers, exact JPEG via {} decoders",
                path.display(),
                jbrd.len(),
                decoders.len()
            );
        }
        fs::remove_file(official).unwrap();
    }
    eprintln!(
        "{checks} independent JPEG reconstruction checks; coefficient frames are official test fixtures, NOT Rust-produced"
    );
}

#[test]
fn fixture_codestream_assembly_validates_part_order_and_termination() {
    let mut complete = SIGNATURE.to_vec();
    complete.extend_from_slice(b"\0\0\0\x0bjxlc\xff\x0a\x07");
    assert_eq!(fixture_codestream(&complete).unwrap(), [0xff, 0x0a, 7]);
    let mut split = SIGNATURE.to_vec();
    split.extend_from_slice(b"\0\0\0\x0ejxlp\0\0\0\0\xff\x0a");
    assert!(fixture_codestream(&split).is_err());
    split.extend_from_slice(b"\0\0\0\x0djxlp\x80\0\0\x01\x07");
    assert_eq!(fixture_codestream(&split).unwrap(), [0xff, 0x0a, 7]);
    split[37] = 2;
    assert!(fixture_codestream(&split).is_err());
    complete.extend_from_slice(b"\0\0\0\x08jxlc");
    assert!(fixture_codestream(&complete).is_err());
}

#[test]
fn reconstruction_box_replacement_preserves_other_boxes_and_rejects_bad_sizes() {
    let mut source = SIGNATURE.to_vec();
    source.extend_from_slice(b"\0\0\0\x0ajbrdAA\0\0\0\0jxlcxyz");
    let result = replace_jbrd(&source, b"BBBB").unwrap();
    let mut expected = SIGNATURE.to_vec();
    expected.extend_from_slice(b"\0\0\0\x0cjbrdBBBB\0\0\0\0jxlcxyz");
    assert_eq!(result, expected);
    assert!(replace_jbrd(&source[..source.len() - 6], b"B").is_err());
    assert!(replace_jbrd(SIGNATURE, b"B").is_err());
    let mut extended = SIGNATURE.to_vec();
    extended.extend_from_slice(b"\0\0\0\x01jbrd\0\0\0\0\0\0\0\x12AA");
    assert!(replace_jbrd(&extended, b"B").is_ok());
    extended.extend_from_slice(b"\0\0\0\x08jbrd");
    assert!(replace_jbrd(&extended, b"B").is_err());
    extended[20..28].copy_from_slice(&u64::MAX.to_be_bytes());
    assert!(replace_jbrd(&extended, b"B").is_err());
}
