use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::Result;

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

fn stdio(path: &Path) -> bool {
    path == Path::new("-")
}

pub(crate) fn check_paths(input: &Path, output: &Path, force: bool) -> Result<()> {
    if stdio(output) {
        return Ok(());
    }
    if !stdio(input) && input == output {
        return Err("input and output paths must differ, even with --force".into());
    }
    let output_meta = match fs::symlink_metadata(output) {
        Ok(meta) => Some(meta),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    if let Some(meta) = output_meta {
        if !stdio(input) {
            let input_path = fs::canonicalize(input)?;
            let output_path = fs::canonicalize(output)?;
            if input_path == output_path {
                return Err("input and output refer to the same path, even with --force".into());
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                let input_meta = fs::metadata(input)?;
                if input_meta.dev() == meta.dev() && input_meta.ino() == meta.ino() {
                    return Err("input and output refer to the same file, even with --force".into());
                }
            }
        }
        if !meta.file_type().is_file() {
            return Err("output must be a regular file, not a directory or symlink".into());
        }
        if !force {
            return Err(format!(
                "output '{}' already exists; use --force to replace it",
                output.display()
            )
            .into());
        }
    }
    Ok(())
}

fn read_limited(mut reader: impl Read, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        let count = match reader.read(&mut chunk) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if count == 0 {
            return Ok(bytes);
        }
        if bytes
            .len()
            .checked_add(count)
            .is_none_or(|size| size > limit)
        {
            return Err("input size overflows or exceeds the configured byte limit".into());
        }
        bytes.try_reserve(count)?;
        bytes.extend_from_slice(&chunk[..count]);
    }
}

pub(crate) fn read_input(path: &Path) -> Result<Vec<u8>> {
    if stdio(path) {
        return read_limited(io::stdin().lock(), usize::MAX);
    }
    let file = File::open(path)?;
    read_limited(file, usize::MAX)
}

struct TemporaryOutput {
    path: PathBuf,
    file: Option<File>,
}

impl TemporaryOutput {
    fn create(output: &Path) -> Result<Self> {
        let parent = output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        for _ in 0..128 {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let name = format!(".rs-jxl-{}-{id}.tmp", std::process::id());
            let path = parent.join(&name);
            // Never let our staging name coincide with the requested destination.
            if output
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|output_name| output_name.eq_ignore_ascii_case(&name))
            {
                continue;
            }
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file: Some(file),
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err("could not reserve a temporary output file".into())
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

pub(crate) fn write_output(input: &Path, output: &Path, force: bool, bytes: &[u8]) -> Result<()> {
    if stdio(output) {
        let mut stdout = io::stdout().lock();
        stdout.write_all(bytes)?;
        stdout.flush()?;
        return Ok(());
    }
    check_paths(input, output, force)?;
    let mut temporary = TemporaryOutput::create(output)?;
    let file = temporary
        .file
        .as_mut()
        .ok_or("temporary output file is unavailable")?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(temporary.file.take());
    check_paths(input, output, force)?;
    if force {
        fs::rename(&temporary.path, output)?;
    } else {
        // Publishing via a hard link is atomic and never replaces a racing writer.
        fs::hard_link(&temporary.path, output).map_err(|error| {
            format!(
                "cannot publish '{}' without overwriting it: {error}",
                output.display()
            )
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            for _ in 0..128 {
                let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir()
                    .join(format!("rs-jxl-cli-test-{}-{id}", std::process::id()));
                if fs::create_dir(&path).is_ok() {
                    return Self(path);
                }
            }
            panic!("cannot create test directory");
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn bounded_reads_reject_one_extra_byte() {
        assert_eq!(read_limited(&b"abcd"[..], 4).unwrap(), b"abcd");
        assert!(read_limited(&b"abcde"[..], 4).is_err());
    }

    #[test]
    fn unlimited_reads_allocate_only_for_received_bytes() {
        assert_eq!(read_limited(&b"abcd"[..], usize::MAX).unwrap(), b"abcd");
        assert!(
            read_limited(std::io::empty(), usize::MAX)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn refuses_same_path_and_canonical_alias_even_when_forced() {
        let dir = TestDir::new();
        let input = dir.0.join("input.png");
        fs::write(&input, b"original").unwrap();
        assert!(check_paths(&input, &input, true).is_err());
        assert!(check_paths(&input, &dir.0.join(".").join("input.png"), true).is_err());
        assert_eq!(fs::read(input).unwrap(), b"original");
    }

    #[test]
    fn publishes_without_clobber_and_force_replaces() {
        let dir = TestDir::new();
        let input = dir.0.join("input.png");
        let output = dir.0.join("output.jxl");
        fs::write(&input, b"original").unwrap();
        write_output(&input, &output, false, b"first").unwrap();
        assert!(write_output(&input, &output, false, b"second").is_err());
        assert_eq!(fs::read(&output).unwrap(), b"first");
        write_output(&input, &output, true, b"second").unwrap();
        assert_eq!(fs::read(&output).unwrap(), b"second");
        assert_eq!(fs::read(&input).unwrap(), b"original");
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 2);
    }

    #[test]
    fn forced_replacement_does_not_modify_an_input_hard_link() {
        let dir = TestDir::new();
        let input = dir.0.join("input.png");
        let output = dir.0.join("output.jxl");
        fs::write(&input, b"original").unwrap();
        fs::hard_link(&input, &output).unwrap();
        let _ = write_output(&input, &output, true, b"replacement");
        assert_eq!(fs::read(&input).unwrap(), b"original");
    }
}
