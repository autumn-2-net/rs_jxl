//! Rebuild original JPEG entropy from signed coefficients, never archived scans.
//! This acceptance test does not claim that a JPEG-to-JXL encoder exists yet.

use std::{env, fs, io::Read, path::PathBuf};

use rs_jxl::jpeg::{
    coefficient_reader::{CoefficientReadLimits, read_coefficients_with_limits},
    data::{CodingProcess, JpegRecord},
    reader::ReadLimits,
    scan_writer::{WriteLimits, reconstruct_from_coefficients},
};

const MAX_INPUT: usize = 16 * 1024 * 1024;

fn inputs() -> Vec<PathBuf> {
    let roots = env::var_os("RS_JXL_JPEG_INPUTS").expect("explicit JPEG input path list required");
    let mut files = Vec::new();
    for root in env::split_paths(&roots) {
        if root.is_file() {
            files.push(root.canonicalize().unwrap());
        } else {
            let mut entries = 0;
            for entry in fs::read_dir(&root).unwrap() {
                entries += 1;
                assert!(entries <= 4096, "bounded nonrecursive discovery exceeded");
                let entry = entry.unwrap();
                let path = entry.path();
                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                if entry.file_type().unwrap().is_file()
                    && (ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg"))
                {
                    files.push(path.canonicalize().unwrap());
                }
            }
        }
    }
    files.sort();
    files.dedup();
    assert!(
        !files.is_empty() && files.len() <= 17,
        "select 1..=17 JPEGs explicitly"
    );
    files
}

#[test]
#[ignore = "set RS_JXL_JPEG_INPUTS; bounded real JPEG coefficient-to-byte reconstruction"]
fn real_jpegs_rebuild_every_original_byte_after_removing_archived_entropy() {
    let limits = CoefficientReadLimits {
        structural: ReadLimits {
            max_input_bytes: MAX_INPUT,
            ..Default::default()
        },
        ..Default::default()
    };
    let (mut total, mut progressive, mut restarted) = (0usize, 0usize, 0usize);
    let paths = inputs();
    for path in &paths {
        let mut input = Vec::new();
        fs::File::open(path)
            .unwrap()
            .take(MAX_INPUT as u64 + 1)
            .read_to_end(&mut input)
            .unwrap();
        assert!(input.len() <= MAX_INPUT);
        let mut jpeg = read_coefficients_with_limits(&input, &limits)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        progressive += usize::from(jpeg.coding_process == Some(CodingProcess::Progressive));
        let mut has_restart = false;
        for record in &mut jpeg.records {
            if let JpegRecord::Entropy(entropy) = record {
                has_restart |= !entropy.restart_markers.is_empty();
                // No original entropy can accidentally satisfy the roundtrip.
                entropy.bytes.clear();
                entropy.bytes.shrink_to_fit();
            }
        }
        restarted += usize::from(has_restart);
        assert_ne!(
            jpeg.reconstruct().unwrap(),
            input,
            "archive-only replay must fail"
        );
        let started = std::time::Instant::now();
        let output = reconstruct_from_coefficients(
            &jpeg,
            &WriteLimits {
                max_output_bytes: input.len(),
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let difference = output.iter().zip(&input).position(|(a, b)| a != b);
        assert!(
            difference.is_none() && output.len() == input.len(),
            "{}: first difference={difference:?}, output={}, input={}",
            path.display(),
            output.len(),
            input.len()
        );
        total += input.len();
        eprintln!(
            "{}: {} bytes reconstructed exactly, {:.3}s; original entropy absent",
            path.display(),
            output.len(),
            started.elapsed().as_secs_f64()
        );
    }
    eprintln!(
        "{} JPEGs, {progressive} progressive, {restarted} restarted, {total} original bytes rebuilt; NOT JXL coding",
        paths.len()
    );
}
