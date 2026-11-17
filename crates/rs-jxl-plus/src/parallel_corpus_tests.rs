//! Fixed validation256 parallel-band size and exact-reconstruction experiment.
//! Diagnostic wall times do not constitute a formal parallel speed benchmark.
use crate::{
    EncodeOptions, encode_jpeg,
    model::band_tools,
    tool_tests::{baseline, reconstruct},
};
use serde_json::{Value, json};
use std::{io::Write, path::Path, time::Instant};
fn append(output: &mut std::fs::File, row: &Value) {
    serde_json::to_writer(&mut *output, row).unwrap();
    output.write_all(b"\n").unwrap();
    output.flush().unwrap();
}
#[test]
#[ignore = "requires fixed validation256 manifest and a new parallel JSONL destination"]
fn parallel_corpus_sizes() {
    let manifest =
        std::env::var_os("RS_JXL_PLUS_CORPUS_MANIFEST").expect("RS_JXL_PLUS_CORPUS_MANIFEST");
    let destination =
        std::env::var_os("RS_JXL_PLUS_PARALLEL_JSONL").expect("RS_JXL_PLUS_PARALLEL_JSONL");
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let cases: Vec<_> = manifest["cases"]
        .as_array()
        .expect("manifest cases")
        .iter()
        .filter(|c| c["split"] == "validation256")
        .collect();
    assert_eq!(cases.len(), 256);
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
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(destination)
        .expect("parallel JSONL must be new");
    append(
        &mut output,
        &json!({"schema":1,"kind":"header","experiment":"parallel_v7_validation256","cases":256,"recipes":[{"max_bands":4,"threads":4},{"max_bands":8,"threads":8}],"band_rule":"min(MCU rows,max_bands)","timing":"diagnostic wall time only; source reads, assertions and JPEG reconstruction excluded from coefficient decode time; not a formal speed benchmark","sha256":"manifest identity externally verified","cost":"40+140*components+8*bands framing + metadata + actual descriptor + arithmetic streams","resume":false}),
    );
    let options = EncodeOptions::default();
    for case in cases {
        let original = std::fs::read(case["source"].as_str().unwrap()).unwrap();
        if let Some(n) = case["original_bytes"].as_u64() {
            assert_eq!(n, original.len() as u64);
        }
        let preparation = Instant::now();
        let prepared = (|| {
            let encoded = encode_jpeg(&original, &options)?;
            let base = baseline(&encoded)?;
            Ok::<_, crate::Error>((encoded, base))
        })();
        let (public, base) = match prepared {
            Ok(value) => value,
            Err(error) => {
                append(
                    &mut output,
                    &json!({"schema":1,"kind":"failure","phase":"preparation","label":case["label"],"source":case["source"],"sha256":case["sha256"],"error":error.to_string()}),
                );
                panic!("parallel corpus preparation failed: {error}");
            }
        };
        let preparation_seconds = preparation.elapsed().as_secs_f64();
        let source_size = public.len() + 1;
        assert_eq!(
            source_size,
            40 + 140 * base.components.len()
                + 8 * base.streams.len()
                + base.metadata.len()
                + 1
                + base.streams.iter().map(Vec::len).sum::<usize>()
        );
        let rows = base.components[0].height / base.components[0].v_samp;
        for (max_bands, threads) in [(4usize, 4usize), (8, 8)] {
            let bands = rows.min(max_bands);
            let timer = Instant::now();
            let encoded = match band_tools::encode_v7_parallel(
                &base.components,
                bands,
                options.limits.max_output_bytes,
                threads,
            ) {
                Ok(value) => value,
                Err(error) => {
                    append(
                        &mut output,
                        &json!({"schema":1,"kind":"failure","phase":"encode","label":case["label"],"source":case["source"],"sha256":case["sha256"],"bands":bands,"threads":threads,"error":error.to_string()}),
                    );
                    panic!("parallel encode failed: {error}");
                }
            };
            let encode_seconds = timer.elapsed().as_secs_f64();
            let mut decoded = base.components.clone();
            for c in &mut decoded {
                c.coefficients.fill(0);
            }
            let streams: Vec<_> = encoded.streams.iter().map(Vec::as_slice).collect();
            let timer = Instant::now();
            if let Err(error) =
                band_tools::decode_v7_parallel(&mut decoded, &encoded.descriptor, &streams, threads)
            {
                append(
                    &mut output,
                    &json!({"schema":1,"kind":"failure","phase":"decode","label":case["label"],"source":case["source"],"sha256":case["sha256"],"bands":bands,"threads":threads,"error":error.to_string()}),
                );
                panic!("parallel decode failed: {error}");
            }
            let coefficient_decode_seconds = timer.elapsed().as_secs_f64();
            if decoded != base.components {
                append(
                    &mut output,
                    &json!({"schema":1,"kind":"failure","phase":"coefficient_comparison","label":case["label"],"bands":bands}),
                );
                panic!("parallel coefficient mismatch");
            }
            let timer = Instant::now();
            let restored = match reconstruct(&base, decoded, &options.limits) {
                Ok(value) => value,
                Err(error) => {
                    append(
                        &mut output,
                        &json!({"schema":1,"kind":"failure","phase":"jpeg_reconstruction","label":case["label"],"bands":bands,"error":error.to_string()}),
                    );
                    panic!("parallel JPEG reconstruction failed: {error}");
                }
            };
            let jpeg_reconstruction_seconds = timer.elapsed().as_secs_f64();
            if restored != original {
                append(
                    &mut output,
                    &json!({"schema":1,"kind":"failure","phase":"jpeg_comparison","label":case["label"],"bands":bands}),
                );
                panic!("parallel original JPEG mismatch");
            }
            let framing = 40 + 140 * base.components.len() + 8 * bands;
            let stream_bytes = encoded.streams.iter().map(Vec::len).sum::<usize>();
            let total = framing + base.metadata.len() + encoded.descriptor.len() + stream_bytes;
            let sampling: Vec<_> = base
                .components
                .iter()
                .map(|c| json!({"h":c.h_samp,"v":c.v_samp}))
                .collect();
            append(
                &mut output,
                &json!({"schema":1,"kind":"case","status":"encoded","label":case["label"],"source":case["source"],"sha256":case["sha256"],"split":"validation256","original_bytes":original.len(),"width":base.width,"height":base.height,"pixels":u64::from(base.width)*u64::from(base.height),"component_sampling":sampling,"source_bands":base.streams.len(),"source_v7_candidate_bytes":source_size,"max_bands":max_bands,"bands":bands,"threads":threads,"framing_bytes":framing,"band_length_bytes":8*bands,"descriptor_bytes":encoded.descriptor.len(),"stream_bytes":stream_bytes,"reconstruction_bytes":base.metadata.len(),"equivalent_complete_bytes":total,"delta_vs_source_v7_bytes":total as i64-source_size as i64,"preparation_seconds":preparation_seconds,"encode_seconds":encode_seconds,"coefficient_decode_seconds":coefficient_decode_seconds,"jpeg_reconstruction_seconds":jpeg_reconstruction_seconds,"coefficients_verified":true,"original_jpeg_bytes_verified":true}),
            );
            eprintln!(
                "parallel validation256 {} bands={bands} threads={threads} bytes={total}",
                case["label"]
            );
        }
    }
    append(
        &mut output,
        &json!({"schema":1,"kind":"complete","cases":256,"verified_outputs":512}),
    );
}
