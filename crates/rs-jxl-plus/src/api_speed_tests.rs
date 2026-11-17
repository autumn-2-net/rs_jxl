//! Public API timing only; never equates Plus verified encoding with bare JXL encoding.
use crate::{EncodeOptions, decode_jpeg, encode_jpeg};
use rs_jxl::{
    decode::{DecodeOptions, reconstruct_jpeg},
    jpeg::{TranscodeOptions, transcode},
};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
const NAMES: [&str; 3] = ["standard_jxl_e3", "standard_jxl_e9", "plus_verified_encode"];
struct Measurement {
    bytes: usize,
    encode: f64,
    decode: f64,
}
fn measure(input: &[u8], recipe: usize) -> Measurement {
    let plus = EncodeOptions {
        metadata_quality: 6,
        ..EncodeOptions::default()
    };
    let standard = TranscodeOptions {
        effort: if recipe == 0 { 3 } else { 9 },
        threads: 1,
        reconstruction_quality: Some(6),
        ..TranscodeOptions::default()
    };
    let timer = Instant::now();
    let encoded = if recipe == 2 {
        encode_jpeg(input, &plus).unwrap()
    } else {
        transcode(input, &standard).unwrap()
    };
    let encode = timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let restored = if recipe == 2 {
        decode_jpeg(&encoded, &plus.limits).unwrap()
    } else {
        reconstruct_jpeg(&encoded, &DecodeOptions::default()).unwrap()
    };
    let decode = timer.elapsed().as_secs_f64();
    assert_eq!(restored, input, "{} original JPEG bytes", NAMES[recipe]);
    Measurement {
        bytes: encoded.len(),
        encode,
        decode,
    }
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
#[test]
#[ignore = "requires explicit existing seven-input manifest and API speed receipt"]
fn public_jpeg_api_speed() {
    let manifest =
        std::env::var_os("RS_JXL_PLUS_SPEED_MANIFEST").expect("RS_JXL_PLUS_SPEED_MANIFEST");
    let destination =
        std::env::var_os("RS_JXL_PLUS_API_SPEED_RECEIPT").expect("RS_JXL_PLUS_API_SPEED_RECEIPT");
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7);
    // Read all sources before any measured API call. Only the fixed old seven
    // are authorized for this focused API timing comparison.
    let inputs: Vec<_> = cases
        .iter()
        .map(|case| {
            let bytes = std::fs::read(case["source"].as_str().unwrap()).unwrap();
            if let Some(n) = case["original_bytes"].as_u64() {
                assert_eq!(bytes.len() as u64, n);
            }
            bytes
        })
        .collect();
    let mut rounds = Vec::new();
    for round in 0..4 {
        // Warmup plus three repetitions; each measured round gives every
        // recipe a different position to reduce fixed order bias.
        for position in 0..3 {
            let recipe = (position + round) % 3;
            for (case_index, input) in inputs.iter().enumerate() {
                let m = measure(input, recipe);
                if round > 0 {
                    rounds.push(json!({"round":round-1,"position":position,"case_index":case_index,"label":cases[case_index]["label"],"recipe":NAMES[recipe],"complete_bytes":m.bytes,"encode_seconds":m.encode,"decode_seconds":m.decode,"standard_validated_pipeline_seconds":if recipe<2 {Some(m.encode+m.decode)}else{None},"plus_verified_encode_seconds":if recipe==2 {Some(m.encode)}else{None},"original_jpeg_bytes_verified":true}));
                }
            }
            eprintln!("API speed round={round} recipe={} complete", NAMES[recipe]);
        }
    }
    let mut per_case = Vec::new();
    for (case_index, case) in cases.iter().enumerate() {
        let mut variants = Vec::new();
        for (recipe, name) in NAMES.iter().enumerate() {
            let samples: Vec<_> = rounds
                .iter()
                .filter(|r| r["case_index"] == case_index && r["recipe"] == *name)
                .collect();
            assert_eq!(samples.len(), 3);
            let size = samples[0]["complete_bytes"].as_u64().unwrap();
            assert!(
                samples
                    .iter()
                    .all(|r| r["complete_bytes"].as_u64() == Some(size))
            );
            variants.push(json!({"recipe":name,"complete_bytes":size,"median_encode_seconds":median(samples.iter().map(|r|r["encode_seconds"].as_f64().unwrap()).collect()),"median_decode_seconds":median(samples.iter().map(|r|r["decode_seconds"].as_f64().unwrap()).collect()),"median_validated_pipeline_or_verified_encode_seconds":median(samples.iter().map(|r|r[if recipe<2 {"standard_validated_pipeline_seconds"}else{"plus_verified_encode_seconds"}].as_f64().unwrap()).collect())}));
        }
        per_case.push(json!({"label":case["label"],"source":case["source"],"manifest_sha256":case["sha256"],"original_bytes":inputs[case_index].len(),"variants":variants}));
    }
    let totals:Vec<_>=NAMES.iter().enumerate().map(|(recipe,name)| {
        let round_totals:Vec<_>=(0..3).map(|round| {
            let rows:Vec<_>=rounds.iter().filter(|r|r["round"]==round && r["recipe"]==*name).collect();
            assert_eq!(rows.len(),7);
            let encode: f64=rows.iter().map(|r|r["encode_seconds"].as_f64().unwrap()).sum();
            let decode: f64=rows.iter().map(|r|r["decode_seconds"].as_f64().unwrap()).sum();
            json!({"round":round,"encode_seconds":encode,"decode_seconds":decode,"validated_pipeline_or_verified_encode_seconds":if recipe<2 {encode+decode}else{encode}})
        }).collect();
        json!({"recipe":name,"complete_bytes":per_case.iter().map(|r|r["variants"][recipe]["complete_bytes"].as_u64().unwrap()).sum::<u64>(),"round_totals":round_totals,"median_encode_seconds":median(round_totals.iter().map(|r|r["encode_seconds"].as_f64().unwrap()).collect()),"median_decode_seconds":median(round_totals.iter().map(|r|r["decode_seconds"].as_f64().unwrap()).collect()),"median_validated_pipeline_or_verified_encode_seconds":median(round_totals.iter().map(|r|r["validated_pipeline_or_verified_encode_seconds"].as_f64().unwrap()).collect())})
    }).collect();
    let receipt = json!({"schema":1,"experiment":"public_jpeg_api_speed","input_set":"existing seven development smoke inputs; no holdout claim","threads":1,"reconstruction_metadata_brotli_quality":6,"warmup_rounds":1,"measured_rounds":3,"recipe_order":"rotate by round","standard_encode_api":"rs_jxl::jpeg::transcode, efforts3/9, threads1, reconstruction_qualitySome6","standard_decode_api":"rs_jxl::decode::reconstruct_jpeg, DecodeOptions::default threads1","standard_decoder_backend":"vendored official jxl-rs Rust decoder coefficient capture + rs-jxl original JPEG reconstruction","plus_encode_api":"crate::encode_jpeg includes complete decode and byte verification","plus_decode_api":"crate::decode_jpeg native v7 arithmetic + shared original JPEG reconstruction","comparison":"standard validated pipeline = bare public transcode + public reconstruct; Plus verified_encode = encode_jpeg alone; separately measured Plus decode is NOT added to this comparison","timing_scope":"API wall time including allocations and pool creation; input reads and byte comparisons excluded; diagnostic local timing, not a formal benchmark","rounds":rounds,"cases":per_case,"totals":totals});
    let destination = Path::new(&destination);
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(destination, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
}
