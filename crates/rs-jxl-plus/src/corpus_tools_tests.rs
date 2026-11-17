//! Fixed-recipe corpus validation; ignored and explicitly opted in via env.
//! Experimental descriptors retain all model, map, framing and metadata costs.
use crate::{
    EncodeOptions, Error, Result, encode_jpeg,
    model::{self, band_tools, context_refine},
    tool_tests::{self, baseline, reconstruct},
};
use rs_jxl::jpeg::{
    cfl::CflFitPolicy,
    coefficient_reader::{CoefficientReadLimits, read_coefficients_with_limits},
    reader::ReadLimits,
    transcode_layout::TranscodeLayout,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Write, path::Path, time::Instant};
const RECIPES: [&str; 6] = [
    "v7_source",
    "v7_bands1",
    "refined2_bands1",
    "refined3_bands1",
    "cfl_source_refined2_bands1",
    "cfl_source_refined3_bands1",
];
struct Payload {
    descriptor: Vec<u8>,
    streams: Vec<Vec<u8>>,
}
fn invalid() -> Error {
    Error::Invalid("corpus tool descriptor")
}
fn compose(maps: &[u8], inner: Payload) -> Result<Payload> {
    let mut descriptor = vec![6, 1];
    for n in [maps.len(), inner.descriptor.len()] {
        descriptor.extend(u32::try_from(n).map_err(|_| invalid())?.to_le_bytes());
    }
    descriptor.extend(maps);
    descriptor.extend(inner.descriptor);
    Ok(Payload {
        descriptor,
        streams: inner.streams,
    })
}
fn decode(cs: &mut [model::Component], payload: &Payload) -> Result<()> {
    let streams: Vec<_> = payload.streams.iter().map(Vec::as_slice).collect();
    let d = &payload.descriptor;
    match d.first() {
        Some(0) if d.len() == 1 => model::decode_strict(cs, &streams),
        Some(4) => band_tools::decode_candidate(cs, d, &streams),
        Some(5) => context_refine::decode_refined(cs, d, &streams),
        Some(6) => {
            if d.len() < 10 || d[1] != 1 {
                return Err(invalid());
            }
            let a = u32::from_le_bytes(d[2..6].try_into().unwrap()) as usize;
            let b = u32::from_le_bytes(d[6..10].try_into().unwrap()) as usize;
            let mid = 10usize.checked_add(a).ok_or_else(invalid)?;
            if a == 0 || b == 0 || mid.checked_add(b) != Some(d.len()) {
                return Err(invalid());
            }
            context_refine::decode_refined(cs, &d[mid..], &streams)?;
            let maps = tool_tests::decode_maps(&d[10..mid], cs)?;
            tool_tests::inverse(cs, &maps)
        }
        _ => Err(invalid()),
    }
}
fn read_progress(path: &Path) -> Vec<Value> {
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    let mut valid = 0usize;
    for line in bytes.split_inclusive(|b| *b == b'\n') {
        if !line.ends_with(b"\n") {
            break;
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            valid += line.len();
            continue;
        }
        rows.push(serde_json::from_slice(line).expect("invalid completed corpus JSONL row"));
        valid += line.len();
    }
    // Only an interrupted final append is discarded, never a completed row.
    if valid != bytes.len() {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(valid as u64)
            .unwrap();
    }
    rows
}
fn same_case(row: &Value, case: &Value, run_id: &str) -> bool {
    ["label", "source", "split", "sha256"]
        .iter()
        .all(|key| row[*key] == case[*key])
        && row["kind"] == "case"
        && row["schema"] == 1
        && row["recipes"] == json!(RECIPES)
        && row["run_id"] == run_id
}
fn append(output: &mut std::fs::File, row: &Value) {
    serde_json::to_writer(&mut *output, row).unwrap();
    output.write_all(b"\n").unwrap();
    output.flush().unwrap();
}
fn summarize(rows: &[Value]) -> Value {
    let mut splits: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for row in rows {
        splits
            .entry(row["split"].as_str().unwrap().to_owned())
            .or_default()
            .push(row);
    }
    let groups:Vec<_>=splits.into_iter().map(|(split,rows)| {
        let skipped:Vec<_>=rows.iter().filter(|r|r["status"]=="skipped").map(|r|json!({"label":r["label"],"reason":r["reason"]})).collect();
        let rows:Vec<_>=rows.into_iter().filter(|r|r["status"]!="skipped").collect();
        let baseline: u64=rows.iter().map(|r|r["variants"][0]["equivalent_complete_bytes"].as_u64().unwrap()).sum();
        let original:u64=rows.iter().map(|r|r["original_bytes"].as_u64().unwrap()).sum();
        let fixed:Vec<_>=RECIPES.iter().map(|name| {
            let mut applicable=0; let mut total=0u64; let mut reference=0u64; let mut fallback=0u64;
            let(mut enc,mut dec,mut rebuild)=(0.0,0.0,0.0);
            for row in &rows {
                let base=row["variants"][0]["equivalent_complete_bytes"].as_u64().unwrap();
                let v=row["variants"].as_array().unwrap().iter().find(|v|v["tool"]==*name).unwrap();
                if v["status"]=="encoded" {
                    applicable+=1; let size=v["equivalent_complete_bytes"].as_u64().unwrap(); total+=size; fallback+=size;reference+=base;
                    enc+=v["encode_seconds"].as_f64().unwrap(); dec+=v["coefficient_decode_seconds"].as_f64().unwrap();rebuild+=v["jpeg_reconstruction_seconds"].as_f64().unwrap();
                } else {fallback+=base;}
            }
            json!({"tool":name,"applicable_cases":applicable,"total_cases":rows.len(),"applicable_complete_bytes":total,"applicable_baseline_bytes":reference,"fixed_policy_with_v7_fallback_bytes":fallback,"encode_seconds_on_applicable":enc,"coefficient_decode_seconds_on_applicable":dec,"jpeg_reconstruction_seconds_on_applicable":rebuild})
        }).collect();
        let oracle:u64=rows.iter().map(|r|r["oracle_complete_bytes"].as_u64().unwrap()).sum();
        json!({"split":split,"encoded_cases":rows.len(),"skipped_cases":skipped,"original_bytes_on_encoded_cases":original,"baseline_complete_bytes":baseline,"fixed_candidates":fixed,"per_file_oracle_complete_bytes":oracle,"oracle_is_measured_minimum_not_a_fixed_encoder_policy":true})
    }).collect();
    json!({"schema":1,"experiment":"fixed_corpus_tools","recipes":RECIPES,"groups":groups,"timing":"diagnostic wall time, not isolated benchmark; decode excludes assertions and JPEG reconstruction", "sha256":"manifest identity only; not independently hashed by this harness", "syntax":"internal experimental equivalent complete sizes, not final public jxl_plus syntax"})
}
#[test]
#[ignore = "requires explicit corpus manifest, JSONL progress and summary destinations"]
fn fixed_tools_corpus() {
    let manifest_path =
        std::env::var_os("RS_JXL_PLUS_CORPUS_MANIFEST").expect("RS_JXL_PLUS_CORPUS_MANIFEST");
    let progress_path =
        std::env::var_os("RS_JXL_PLUS_CORPUS_JSONL").expect("RS_JXL_PLUS_CORPUS_JSONL");
    let receipt_path =
        std::env::var_os("RS_JXL_PLUS_CORPUS_RECEIPT").expect("RS_JXL_PLUS_CORPUS_RECEIPT");
    // The runner supplies a source/build fingerprint, not just recipe names.
    let run_id = std::env::var("RS_JXL_PLUS_CORPUS_RUN_ID").expect("RS_JXL_PLUS_CORPUS_RUN_ID");
    assert!(!run_id.is_empty());
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().expect("manifest cases");
    for (i, c) in cases.iter().enumerate() {
        for key in ["label", "source", "split"] {
            assert!(!c[key].as_str().expect("case string").is_empty());
        }
        assert!(
            !cases[..i]
                .iter()
                .any(|p| p["label"] == c["label"] && p["split"] == c["split"]),
            "duplicate label/split"
        );
    }
    let progress_path = Path::new(&progress_path);
    if let Some(parent) = progress_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let previous = read_progress(progress_path);
    for row in &previous {
        assert_eq!(
            row["run_id"], run_id,
            "progress belongs to a different implementation; use a fresh JSONL path"
        );
    }
    let mut completed = Vec::new();
    let mut output = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(progress_path)
        .unwrap();
    let options = EncodeOptions::default();
    for case in cases {
        if let Some(row) = previous.iter().rev().find(|r| same_case(r, case, &run_id)) {
            assert_eq!(
                std::fs::metadata(case["source"].as_str().unwrap())
                    .unwrap()
                    .len(),
                row["original_bytes"].as_u64().unwrap(),
                "resumed source size changed"
            );
            completed.push(row.clone());
            continue;
        }
        let original = std::fs::read(case["source"].as_str().unwrap()).unwrap();
        if let Some(n) = case["original_bytes"].as_u64() {
            assert_eq!(n, original.len() as u64);
        }
        // Metadata preparation and baseline extraction are outside recipe times.
        let prep = Instant::now();
        let encoded = match encode_jpeg(&original, &options) {
            Ok(value) => value,
            Err(error @ Error::Internal(_)) => {
                append(
                    &mut output,
                    &json!({"schema":1,"recipes":RECIPES,"kind":"failure","label":case["label"],"source":case["source"],"split":case["split"],"phase":"public_encode","error":error.to_string()}),
                );
                panic!("public encoder internal failure: {error}");
            }
            Err(error) => {
                let row = json!({"schema":1,"run_id":run_id,"recipes":RECIPES,"kind":"case","status":"skipped","label":case["label"],"source":case["source"],"split":case["split"],"sha256":case["sha256"],"original_bytes":original.len(),"reason":error.to_string(),"preparation_seconds":prep.elapsed().as_secs_f64()});
                append(&mut output, &row);
                completed.push(row);
                continue;
            }
        };
        let base = baseline(&encoded).unwrap();
        let prep_seconds = prep.elapsed().as_secs_f64();
        let jpeg = read_coefficients_with_limits(
            &original,
            &CoefficientReadLimits {
                structural: ReadLimits {
                    max_input_bytes: options.limits.max_input_bytes,
                    max_scans: options.limits.max_scans,
                    max_records: options.limits.max_hint_entries,
                    max_restart_markers: options.limits.max_hint_entries,
                    ..ReadLimits::default()
                },
                max_coefficient_bytes: options.limits.max_coefficient_bytes,
                max_block_visits: options.limits.max_block_visits,
                max_hint_entries: options.limits.max_hint_entries,
                max_padding_bits: options.limits.max_metadata_bytes.saturating_mul(8),
                ..CoefficientReadLimits::default()
            },
        )
        .unwrap();
        let eligible = TranscodeLayout::new(&jpeg).is_ok_and(|layout| layout.can_use_ac_cfl());
        let mut variants = Vec::new();
        for (index, name) in RECIPES.iter().enumerate() {
            if index >= 4 && !eligible {
                variants.push(json!({"tool":name,"status":"not_applicable","reason":"Source CfL requires supported 4:4:4 layout"}));
                continue;
            }
            let timer = Instant::now();
            let attempt = (|| -> Result<Payload> {
                match index {
                    0 => Ok(Payload {
                        descriptor: vec![0],
                        streams: model::encode(
                            &base.components,
                            options.limits.max_output_bytes - 1,
                        )?,
                    }),
                    1 => {
                        let e = band_tools::encode_candidate(
                            &base.components,
                            band_tools::Tool::V7,
                            1,
                            options.limits.max_output_bytes,
                        )?;
                        Ok(Payload {
                            descriptor: e.descriptor,
                            streams: e.streams,
                        })
                    }
                    2 | 3 => {
                        let e = context_refine::encode_refined(
                            &base.components,
                            index as u8,
                            1,
                            options.limits.max_output_bytes,
                        )?;
                        Ok(Payload {
                            descriptor: e.descriptor,
                            streams: e.streams,
                        })
                    }
                    _ => {
                        let (residual, maps) =
                            tool_tests::forward(&jpeg, &base.components, CflFitPolicy::Source)?;
                        let maps = tool_tests::encode_maps(&maps, options.limits.max_output_bytes)?;
                        let cap = options
                            .limits
                            .max_output_bytes
                            .checked_sub(maps.len() + 10)
                            .ok_or(Error::Limit("corpus composition"))?;
                        let e =
                            context_refine::encode_refined(&residual, (index - 2) as u8, 1, cap)?;
                        compose(
                            &maps,
                            Payload {
                                descriptor: e.descriptor,
                                streams: e.streams,
                            },
                        )
                    }
                }
            })();
            let encode_seconds = timer.elapsed().as_secs_f64();
            let payload = match attempt {
                Ok(v) => v,
                Err(error) => {
                    if matches!(error, Error::Internal(_)) {
                        append(
                            &mut output,
                            &json!({"kind":"failure","run_id":run_id,"label":case["label"],"tool":name,"phase":"encode","error":error.to_string()}),
                        );
                        panic!("{name}: internal failure: {error}");
                    }
                    assert!(index >= 4, "required fixed recipe {name} failed: {error}");
                    variants.push(json!({"tool":name,"status":"rejected","reason":error.to_string(),"encode_seconds":encode_seconds}));
                    continue;
                }
            };
            if index == 0 {
                assert_eq!(payload.streams, base.streams);
            }
            let mut decoded = base.components.clone();
            for c in &mut decoded {
                c.coefficients.fill(0);
            }
            let timer = Instant::now();
            if let Err(error) = decode(&mut decoded, &payload) {
                append(
                    &mut output,
                    &json!({"schema":1,"recipes":RECIPES,"kind":"failure","label":case["label"],"source":case["source"],"split":case["split"],"tool":name,"phase":"coefficient_decode","error":error.to_string()}),
                );
                panic!("{name}: coefficient decode failed: {error}");
            }
            let coefficient_decode_seconds = timer.elapsed().as_secs_f64();
            if decoded != base.components {
                append(
                    &mut output,
                    &json!({"schema":1,"recipes":RECIPES,"kind":"failure","label":case["label"],"tool":name,"phase":"coefficient_comparison"}),
                );
                panic!("{name}: coefficient mismatch");
            }
            let timer = Instant::now();
            let rebuilt = match reconstruct(&base, decoded, &options.limits) {
                Ok(value) => value,
                Err(error) => {
                    append(
                        &mut output,
                        &json!({"schema":1,"recipes":RECIPES,"kind":"failure","label":case["label"],"tool":name,"phase":"jpeg_reconstruction","error":error.to_string()}),
                    );
                    panic!("{name}: JPEG reconstruction failed: {error}");
                }
            };
            let jpeg_reconstruction_seconds = timer.elapsed().as_secs_f64();
            if rebuilt != original {
                append(
                    &mut output,
                    &json!({"schema":1,"recipes":RECIPES,"kind":"failure","label":case["label"],"tool":name,"phase":"jpeg_comparison"}),
                );
                panic!("{name}: original JPEG mismatch");
            }
            let framing = 40 + 140 * base.components.len() + 8 * payload.streams.len();
            let stream_bytes = payload.streams.iter().map(Vec::len).sum::<usize>();
            let total = framing + base.metadata.len() + payload.descriptor.len() + stream_bytes;
            if index == 0 {
                assert_eq!(total, encoded.len() + 1, "source V7 complete size");
            }
            variants.push(json!({"tool":name,"status":"encoded","bands":payload.streams.len(),"descriptor_bytes":payload.descriptor.len(),"stream_bytes":stream_bytes,"framing_bytes":framing,"band_length_bytes":8*payload.streams.len(),"reconstruction_bytes":base.metadata.len(),"equivalent_complete_bytes":total,"encode_seconds":encode_seconds,"coefficient_decode_seconds":coefficient_decode_seconds,"jpeg_reconstruction_seconds":jpeg_reconstruction_seconds,"coefficients_verified":true,"original_jpeg_bytes_verified":true}));
        }
        let winner = variants
            .iter()
            .filter(|v| v["status"] == "encoded")
            .min_by_key(|v| v["equivalent_complete_bytes"].as_u64().unwrap())
            .unwrap();
        let sampling: Vec<_> = base
            .components
            .iter()
            .map(|c| json!({"h":c.h_samp,"v":c.v_samp}))
            .collect();
        let row = json!({"schema":1,"run_id":run_id,"recipes":RECIPES,"kind":"case","status":"encoded","label":case["label"],"source":case["source"],"split":case["split"],"sha256":case["sha256"],"original_bytes":original.len(),"width":base.width,"height":base.height,"pixels":u64::from(base.width)*u64::from(base.height),"component_sampling":sampling,"source_bands":base.streams.len(),"cfl_eligible":eligible,"preparation_seconds":prep_seconds,"oracle_tool":winner["tool"],"oracle_complete_bytes":winner["equivalent_complete_bytes"],"oracle_not_fixed_policy":true,"variants":variants});
        append(&mut output, &row);
        eprintln!("corpus {} {} complete", case["split"], case["label"]);
        completed.push(row);
    }
    let receipt_path = Path::new(&receipt_path);
    if let Some(parent) = receipt_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(
        receipt_path,
        serde_json::to_vec_pretty(&summarize(&completed)).unwrap(),
    )
    .unwrap();
}
