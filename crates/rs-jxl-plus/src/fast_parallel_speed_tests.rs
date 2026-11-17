//! Opt-in decoding speed / complete-size tradeoffs; fixed model profiles.
use crate::{
    EncodeOptions, encode_jpeg,
    model::band_tools,
    tool_tests::{baseline, reconstruct},
};
use serde_json::{Value, json};
use std::time::Instant;

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "requires speed manifest/receipt; run without competing codec or build work"]
fn fast_parallel_tradeoffs() {
    let path = std::env::var_os("RS_JXL_PLUS_SPEED_MANIFEST").unwrap();
    let destination = std::env::var_os("RS_JXL_PLUS_PARALLEL_SPEED_RECEIPT").unwrap();
    let manifest: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let options = EncodeOptions::default();
    let mut cases = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let original = std::fs::read(case["source"].as_str().unwrap()).unwrap();
        let public = encode_jpeg(&original, &options).unwrap();
        let base = baseline(&public).unwrap();
        let rows = base.components[0].height / base.components[0].v_samp;
        let source = base.streams.len();
        let recipes = [
            ("v7_bands1_threads1", band_tools::Tool::V7, 1, 1),
            ("v7_source_threads4", band_tools::Tool::V7, source, 4),
            ("v7_bands4_threads4", band_tools::Tool::V7, rows.min(4), 4),
            ("v7_bands8_threads8", band_tools::Tool::V7, rows.min(8), 8),
            ("fast_source_threads1", band_tools::Tool::Fast, source, 1),
        ];
        let mut enc: [Vec<f64>; 5] = Default::default();
        let mut dec: [Vec<f64>; 5] = Default::default();
        let mut sizes = [0usize; 5];
        for round in 0..4 {
            for step in 0..5 {
                let i = (step + round) % 5;
                let (name, tool, bands, threads) = recipes[i];
                let t = Instant::now();
                let payload = if threads == 1 {
                    band_tools::encode_candidate(
                        &base.components,
                        tool,
                        bands,
                        options.limits.max_output_bytes,
                    )
                    .unwrap()
                } else {
                    band_tools::encode_v7_parallel(
                        &base.components,
                        bands,
                        options.limits.max_output_bytes,
                        threads,
                    )
                    .unwrap()
                };
                let et = t.elapsed().as_secs_f64();
                let size = 40
                    + 140 * base.components.len()
                    + 8 * bands
                    + base.metadata.len()
                    + payload.descriptor.len()
                    + payload.streams.iter().map(Vec::len).sum::<usize>();
                if round == 0 {
                    sizes[i] = size;
                    if threads > 1 {
                        let serial = band_tools::encode_candidate(
                            &base.components,
                            tool,
                            bands,
                            options.limits.max_output_bytes,
                        )
                        .unwrap();
                        assert_eq!(payload.descriptor, serial.descriptor);
                        assert_eq!(
                            payload.streams, serial.streams,
                            "{name}: parallel altered the serial stream"
                        );
                    }
                } else {
                    assert_eq!(sizes[i], size);
                }
                let refs: Vec<_> = payload.streams.iter().map(Vec::as_slice).collect();
                let mut cs = base.components.clone();
                for c in &mut cs {
                    c.coefficients.fill(0);
                }
                let t = Instant::now();
                if threads == 1 {
                    band_tools::decode_candidate(&mut cs, &payload.descriptor, &refs).unwrap();
                } else {
                    band_tools::decode_v7_parallel(&mut cs, &payload.descriptor, &refs, threads)
                        .unwrap();
                }
                let dt = t.elapsed().as_secs_f64();
                assert_eq!(cs, base.components);
                if round == 0 {
                    assert_eq!(reconstruct(&base, cs, &options.limits).unwrap(), original);
                }
                if round > 0 {
                    enc[i].push(et);
                    dec[i].push(dt);
                }
            }
        }
        let variants: Vec<_> = recipes.iter().enumerate().map(|(i,(name,_,bands,threads))|json!({
            "tool":name,"bands":bands,"threads":threads,"equivalent_complete_bytes":sizes[i],
            "encode_seconds":enc[i],"decode_seconds":dec[i],"median_encode_seconds":median(&enc[i]),"median_decode_seconds":median(&dec[i])
        })).collect();
        eprintln!("parallel tradeoff {} complete", case["label"]);
        cases.push(json!({"label":case["label"],"source":case["source"],"original_bytes":original.len(),"original_v7_candidate_bytes":public.len()+1,"variants":variants}));
    }
    std::fs::write(destination, serde_json::to_vec_pretty(&json!({
        "experiment":"fixed_fast_parallel_tradeoffs","warmup_rounds":1,"measured_rounds":3,
        "timing_scope":"coefficient stage, includes per-call thread pool/model/zigzag allocation, excludes JPEG preparation, destination clearing, assertions, JPEG reconstruction and reference serial encode",
        "all_candidates_restore_original_jpeg":true,"all_parallel_streams_match_same_band_serial":true,
        "notes":"Increasing bands changes boundary prediction and model warmup, hence size. Source-band parallelism keeps the original streams. More workers use more CPU cores and concurrently live model state; no single-core algorithm speedup is implied.","cases":cases
    })).unwrap()).unwrap();
}
