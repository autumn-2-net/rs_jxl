//! Development-only real JPEG structural archive checks. No JXL transcoder is
//! exercised here: preserving opaque scan bytes is not coefficient re-encoding.

use rs_jxl::jpeg::data::CodingProcess;
use rs_jxl::jpeg::reader::read_jpeg;
use std::path::{Path, PathBuf};

fn collect(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path)
            .expect("JPEG fixture directory")
            .map(|entry| entry.unwrap().path())
            .collect();
        entries.sort();
        for entry in entries {
            // Do not recurse through directory symlinks or junctions.
            if std::fs::symlink_metadata(&entry)
                .unwrap()
                .file_type()
                .is_symlink()
            {
                continue;
            }
            collect(&entry, files);
        }
    } else if path.extension().is_some_and(|extension| {
        extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg")
    }) {
        files.push(path.to_path_buf());
    }
}

#[test]
#[ignore = "set RS_JXL_JPEG_INPUTS to real JPEG files/directories (OS path-list separator)"]
fn real_jpeg_marker_scan_archives_reproduce_every_original_byte() {
    let inputs = std::env::var_os("RS_JXL_JPEG_INPUTS").expect("RS_JXL_JPEG_INPUTS is required");
    let mut files = Vec::new();
    for path in std::env::split_paths(&inputs) {
        assert!(path.exists(), "missing fixture: {}", path.display());
        collect(&path, &mut files);
    }
    files.sort();
    files.dedup();
    assert!(!files.is_empty(), "no JPEG fixtures found");
    let mut progressive = 0usize;
    let mut total_bytes = 0usize;
    for path in &files {
        let input = std::fs::read(path).unwrap();
        let parsed =
            read_jpeg(&input).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        assert_eq!(parsed.reconstruct().unwrap(), input, "{}", path.display());
        assert!(parsed.width != 0 && parsed.height != 0);
        assert!(!parsed.scan_info.is_empty());
        assert!(
            parsed
                .components
                .iter()
                .all(|component| component.coefficients.is_empty())
        );
        for scan in 0..parsed.scan_info.len() {
            assert!(parsed.entropy_data(scan).is_some());
        }
        progressive += usize::from(parsed.coding_process == Some(CodingProcess::Progressive));
        total_bytes += input.len();
        eprintln!(
            "{}: {}x{}, {} scans, {} bytes exact",
            path.display(),
            parsed.width,
            parsed.height,
            parsed.scan_info.len(),
            input.len()
        );
    }
    eprintln!(
        "{} JPEG structural archives, {} progressive, {} bytes; NOT a JXL transcode test",
        files.len(),
        progressive,
        total_bytes
    );
}
