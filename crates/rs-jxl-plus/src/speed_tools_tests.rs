//! Opt-in, single-thread coefficient-stage timing and exact pre/post snapshots.
//! File I/O, JPEG preparation, destination clearing and verification are untimed.
use crate::{
    EncodeOptions, encode_jpeg,
    model::{self, band_tools, context_refine},
    tool_tests::{baseline, reconstruct},
};
use serde_json::{Value, json};
use std::{hint::black_box, path::PathBuf, time::Instant};

fn encode(
    cs: &[model::Component],
    profile: u8,
    source_bands: usize,
    cap: usize,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    match profile {
        0 => (vec![0], model::encode(cs, cap).unwrap()),
        1 => {
            let e = band_tools::encode_candidate(cs, band_tools::Tool::V7, 1, cap).unwrap();
            (e.descriptor, e.streams)
        }
        2 | 3 => {
            let e = context_refine::encode_refined(cs, profile, 1, cap).unwrap();
            (e.descriptor, e.streams)
        }
        _ => unreachable!("profile {profile}, source bands {source_bands}"),
    }
}
fn decode(cs: &mut [model::Component], descriptor: &[u8], streams: &[Vec<u8>]) {
    let refs: Vec<_> = streams.iter().map(Vec::as_slice).collect();
    match descriptor[0] {
        0 => model::decode_strict(cs, &refs).unwrap(),
        4 => band_tools::decode_candidate(cs, descriptor, &refs).unwrap(),
        5 => context_refine::decode_refined(cs, descriptor, &refs).unwrap(),
        _ => unreachable!(),
    }
}
fn snapshot(descriptor: &[u8], streams: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend((descriptor.len() as u64).to_le_bytes());
    bytes.extend(descriptor);
    bytes.extend((streams.len() as u64).to_le_bytes());
    for stream in streams {
        bytes.extend((stream.len() as u64).to_le_bytes());
        bytes.extend(stream);
    }
    bytes
}
fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "explicit manifest, receipt and snapshot directory required; run without competing codec work"]
fn coefficient_speed() {
    let manifest_path = std::env::var_os("RS_JXL_PLUS_SPEED_MANIFEST").unwrap();
    let receipt = std::env::var_os("RS_JXL_PLUS_SPEED_RECEIPT").unwrap();
    let snapshots = PathBuf::from(std::env::var_os("RS_JXL_PLUS_SPEED_SNAPSHOTS").unwrap());
    let save = std::env::var("RS_JXL_PLUS_SPEED_SAVE").as_deref() == Ok("1");
    let repeats: usize = std::env::var("RS_JXL_PLUS_SPEED_REPEATS")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .unwrap();
    assert!((3..=15).contains(&repeats) && repeats % 2 == 1);
    std::fs::create_dir_all(&snapshots).unwrap();
    let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let options = EncodeOptions::default();
    let mut results = Vec::new();
    for (case_index, case) in manifest["cases"].as_array().unwrap().iter().enumerate() {
        let original = std::fs::read(case["source"].as_str().unwrap()).unwrap();
        let public = encode_jpeg(&original, &options).unwrap();
        let base = baseline(&public).unwrap();
        let mut enc_times: [Vec<f64>; 4] = Default::default();
        let mut dec_times: [Vec<f64>; 4] = Default::default();
        let mut stream_sizes = [0usize; 4];
        for repeat in 0..=repeats {
            // Rotate the order; round zero warms every recipe and saves/checks bytes.
            for step in 0..4 {
                let profile = (step + repeat) % 4;
                let t = Instant::now();
                let (descriptor, streams) = encode(
                    black_box(&base.components),
                    profile as u8,
                    base.streams.len(),
                    options.limits.max_output_bytes,
                );
                let enc = t.elapsed().as_secs_f64();
                if repeat == 0 {
                    let path = snapshots.join(format!("{case_index:04}-{profile}.bin"));
                    let bytes = snapshot(&descriptor, &streams);
                    if save {
                        assert!(!path.exists(), "refusing to overwrite reference snapshot");
                        std::fs::write(path, bytes).unwrap();
                    } else {
                        assert_eq!(
                            std::fs::read(path).unwrap(),
                            bytes,
                            "{} profile{profile}: bitstream changed",
                            case["label"]
                        );
                    }
                }
                stream_sizes[profile] = streams.iter().map(Vec::len).sum();
                let mut cs = base.components.clone();
                for c in &mut cs {
                    c.coefficients.fill(0);
                }
                let t = Instant::now();
                decode(black_box(&mut cs), &descriptor, &streams);
                let dec = t.elapsed().as_secs_f64();
                assert_eq!(cs, base.components);
                if repeat == 0 {
                    assert_eq!(reconstruct(&base, cs, &options.limits).unwrap(), original);
                }
                if repeat != 0 {
                    enc_times[profile].push(enc);
                    dec_times[profile].push(dec);
                }
            }
        }
        let variants: Vec<_> = (0..4).map(|p| json!({
            "tool":(["source_v7", "v7_bands1", "refined2_bands1", "refined3_bands1"][p]),
            "stream_bytes":stream_sizes[p], "encode_seconds":enc_times[p], "decode_seconds":dec_times[p],
            "median_encode_seconds":median(&enc_times[p]), "median_decode_seconds":median(&dec_times[p]),
        })).collect();
        eprintln!(
            "speed {}/{} {}",
            case_index + 1,
            manifest["cases"].as_array().unwrap().len(),
            case["label"]
        );
        results.push(
            json!({"label":case["label"],"source":case["source"],"original_bytes":original.len(),
            "pixels":u64::from(base.width)*u64::from(base.height),"variants":variants}),
        );
    }
    std::fs::write(receipt, serde_json::to_vec_pretty(&json!({
        "experiment":"coefficient_speed_v1", "manifest":PathBuf::from(manifest_path),
        "repeats":repeats,"warmup_rounds":1,"threads":1,"snapshots_saved":save,
        "verified_coefficients":true,"verified_original_jpeg_bytes":true,
        "timing_scope":"Coefficient encode/decode including model allocation, zigzag and descriptor dispatch; excludes file I/O, public JPEG preparation, clearing destination, assertions and JPEG reconstruction.",
        "cases":results
    })).unwrap()).unwrap();
}
