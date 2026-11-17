//! Fixed standard-JXL size/correctness reference for the validation256 split.
//! Timings are diagnostic API wall times, not a formal speed benchmark.
use rs_jxl::{
    decode::{DecodeOptions, reconstruct_jpeg},
    jpeg::{TranscodeOptions, transcode},
};
use serde_json::{Value, json};
use std::{io::Write, path::Path, time::Instant};
fn append(output: &mut std::fs::File, row: &Value) {
    serde_json::to_writer(&mut *output, row).unwrap();
    output.write_all(b"\n").unwrap();
    output.flush().unwrap();
}
#[test]
#[ignore = "requires explicit validation256 manifest and a new standard-JXL JSONL destination"]
fn standard_jxl_corpus() {
    let manifest =
        std::env::var_os("RS_JXL_PLUS_CORPUS_MANIFEST").expect("RS_JXL_PLUS_CORPUS_MANIFEST");
    let destination =
        std::env::var_os("RS_JXL_PLUS_STANDARD_JSONL").expect("RS_JXL_PLUS_STANDARD_JSONL");
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let cases: Vec<_> = manifest["cases"]
        .as_array()
        .expect("manifest cases")
        .iter()
        .filter(|c| c["split"] == "validation256")
        .collect();
    assert_eq!(cases.len(), 256, "fixed validation256 set only");
    for (i, case) in cases.iter().enumerate() {
        for key in ["label", "source", "sha256"] {
            assert!(!case[key].as_str().expect("case identity string").is_empty());
        }
        assert!(
            !cases[..i].iter().any(|p| p["label"] == case["label"]),
            "duplicate validation label"
        );
    }
    let destination = Path::new(&destination);
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    // Deliberately no resume: a preexisting file is rejected rather than
    // accidentally mixing recipes, source revisions or incomplete runs.
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(destination)
        .expect("standard JSONL must be a new file");
    append(
        &mut output,
        &json!({"schema":1,"kind":"header","experiment":"standard_jxl_validation256","cases":256,"efforts":[3,9],"threads":1,"reconstruction_quality":6,"repetitions":1,"resume":false,"timing":"diagnostic API wall time only; reads and byte assertions excluded, not a formal benchmark","decoder_backend":"vendored official jxl-rs coefficient capture plus rs-jxl exact JPEG reconstruction","manifest_sha256":"identity supplied and verified externally; not hashed by this harness"}),
    );
    for case in cases {
        let original = std::fs::read(case["source"].as_str().unwrap()).unwrap();
        if let Some(n) = case["original_bytes"].as_u64() {
            assert_eq!(n, original.len() as u64);
        }
        for effort in [3u8, 9] {
            let options = TranscodeOptions {
                effort,
                threads: 1,
                reconstruction_quality: Some(6),
                ..TranscodeOptions::default()
            };
            let timer = Instant::now();
            let encoded = match transcode(&original, &options) {
                Ok(value) => value,
                Err(error) => {
                    append(
                        &mut output,
                        &json!({"schema":1,"kind":"failure","phase":"encode","label":case["label"],"source":case["source"],"split":"validation256","sha256":case["sha256"],"effort":effort,"error":error.to_string(),"original_bytes":original.len(),"encode_seconds":timer.elapsed().as_secs_f64()}),
                    );
                    panic!(
                        "standard JXL E{effort} encode {} failed: {error}",
                        case["label"]
                    );
                }
            };
            let encode_seconds = timer.elapsed().as_secs_f64();
            let timer = Instant::now();
            let rebuilt = match reconstruct_jpeg(&encoded, &DecodeOptions::default()) {
                Ok(value) => value,
                Err(error) => {
                    append(
                        &mut output,
                        &json!({"schema":1,"kind":"failure","phase":"decode","label":case["label"],"source":case["source"],"split":"validation256","sha256":case["sha256"],"effort":effort,"error":error.to_string(),"complete_bytes":encoded.len(),"encode_seconds":encode_seconds,"decode_seconds":timer.elapsed().as_secs_f64()}),
                    );
                    panic!(
                        "standard JXL E{effort} decode {} failed: {error}",
                        case["label"]
                    );
                }
            };
            let decode_seconds = timer.elapsed().as_secs_f64();
            if rebuilt != original {
                append(
                    &mut output,
                    &json!({"schema":1,"kind":"failure","phase":"byte_comparison","label":case["label"],"source":case["source"],"split":"validation256","sha256":case["sha256"],"effort":effort,"complete_bytes":encoded.len()}),
                );
                panic!(
                    "standard JXL E{effort} original JPEG mismatch {}",
                    case["label"]
                );
            }
            append(
                &mut output,
                &json!({"schema":1,"kind":"case","status":"encoded","label":case["label"],"source":case["source"],"split":"validation256","sha256":case["sha256"],"effort":effort,"threads":1,"reconstruction_quality":6,"original_bytes":original.len(),"complete_bytes":encoded.len(),"encode_seconds":encode_seconds,"decode_seconds":decode_seconds,"original_jpeg_bytes_verified":true}),
            );
            eprintln!(
                "standard validation256 {} E{effort} bytes={}",
                case["label"],
                encoded.len()
            );
        }
    }
    append(
        &mut output,
        &json!({"schema":1,"kind":"complete","cases":256,"verified_outputs":512}),
    );
}
