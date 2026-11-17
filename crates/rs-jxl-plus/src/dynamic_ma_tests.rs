//! Opt-in original-domain v7 + learned dynamic probability-context comparison.
//! All candidates preserve coefficient predictions and original expert updates.
use crate::{
    EncodeOptions, decode_jpeg, encode_jpeg,
    model::ma_context::{ContextMode, decode_context, encode_context, train_context},
    tool_tests::{baseline, reconstruct},
};
use serde_json::json;
use std::time::Instant;

#[test]
#[ignore = "requires the existing seven-input manifest and an explicit receipt destination"]
fn dynamic_ma_context_seven_inputs() {
    let manifest_path = std::env::var_os("RS_JXL_PLUS_DYNAMIC_MA_MANIFEST")
        .expect("RS_JXL_PLUS_DYNAMIC_MA_MANIFEST");
    let receipt_path =
        std::env::var_os("RS_JXL_PLUS_DYNAMIC_MA_RECEIPT").expect("RS_JXL_PLUS_DYNAMIC_MA_RECEIPT");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7, "only the existing seven smoke inputs");
    let options = EncodeOptions::default();
    let mut rows = Vec::new();
    let mut totals = [0usize; 6];
    let (mut original_total, mut baseline_total, mut selected_total) = (0usize, 0usize, 0usize);
    let mut encoded_candidates = 0usize;
    for case in cases {
        let path = case["source"].as_str().unwrap();
        let original = std::fs::read(path).unwrap();
        assert_eq!(Some(original.len() as u64), case["original_bytes"].as_u64());
        let baseline_bytes = encode_jpeg(&original, &options).unwrap();
        assert_eq!(
            decode_jpeg(&baseline_bytes, &options.limits).unwrap(),
            original
        );
        let base = baseline(&baseline_bytes).unwrap();
        let baseline_stream_bytes = base.streams.iter().map(Vec::len).sum::<usize>();
        assert_eq!(
            base.framing + base.metadata.len() + baseline_stream_bytes,
            baseline_bytes.len()
        );
        let start = Instant::now();
        let training = train_context(&base.components).unwrap();
        let training_seconds = start.elapsed().as_secs_f64();
        let baseline_complete = baseline_bytes.len() + 1;
        let mut best = baseline_complete;
        let mut winner = "original_v7".to_owned();
        let mut variants = Vec::new();
        for (mode_index, (mode, name)) in [
            (ContextMode::Single, "single_leaf"),
            (ContextMode::Fixed, "fixed_event_classes"),
            (ContextMode::Learned, "learned_ma_contexts"),
        ]
        .into_iter()
        .enumerate()
        {
            for (rate_index, rate) in [6u8, 8u8].into_iter().enumerate() {
                let start = Instant::now();
                let candidate = encode_context(
                    &base.components,
                    &training,
                    mode,
                    rate,
                    options.limits.max_output_bytes,
                )
                .unwrap();
                let encode_seconds = start.elapsed().as_secs_f64();
                assert_eq!(candidate.accounting.mode, mode_index as u8);
                assert_eq!(candidate.accounting.rate, rate);
                assert_eq!(candidate.streams.len(), base.streams.len());
                assert_eq!(candidate.accounting.modelled_bits, training.modelled_bits);
                assert_eq!(candidate.accounting.direct_bits, training.direct_bits);
                let mut decoded = base.components.clone();
                for c in &mut decoded {
                    c.coefficients.fill(0);
                }
                let start = Instant::now();
                decode_context(
                    &mut decoded,
                    &candidate.descriptor,
                    &candidate
                        .streams
                        .iter()
                        .map(Vec::as_slice)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                assert_eq!(decoded, base.components, "{name} rate={rate}: coefficients");
                assert_eq!(
                    reconstruct(&base, decoded, &options.limits).unwrap(),
                    original,
                    "{name} rate={rate}: original JPEG bytes"
                );
                let decode_verify_seconds = start.elapsed().as_secs_f64();
                let stream_bytes = candidate.streams.iter().map(Vec::len).sum::<usize>();
                let descriptor_bytes = candidate.descriptor.len();
                assert_eq!(candidate.accounting.stream_bytes, stream_bytes);
                assert_eq!(candidate.accounting.descriptor_bytes, descriptor_bytes);
                assert_eq!(
                    candidate.accounting.total_bytes,
                    stream_bytes + descriptor_bytes
                );
                let complete = base.framing + base.metadata.len() + descriptor_bytes + stream_bytes;
                let label = format!("{name}_rate{rate}");
                if complete < best {
                    best = complete;
                    winner = label.clone();
                }
                totals[mode_index * 2 + rate_index] += complete;
                encoded_candidates += 1;
                eprintln!(
                    "Dynamic MA {} {label} baseline={baseline_complete} complete={complete} descriptor={descriptor_bytes} leaves={}",
                    case["label"], candidate.accounting.leaves
                );
                variants.push(json!({
                    "tool":name,"rate":rate,"status":"encoded",
                    "tree_leaves":candidate.accounting.leaves,
                    "probability_bins_per_leaf":candidate.accounting.bins,
                    "descriptor_bytes":descriptor_bytes,"stream_bytes":stream_bytes,
                    "equivalent_complete_bytes":complete,
                    "delta_vs_original_v7_candidate":complete as i64-baseline_complete as i64,
                    "modelled_bits":candidate.accounting.modelled_bits,
                    "direct_bits":candidate.accounting.direct_bits,
                    "decoded_coefficients_verified":true,"original_jpeg_bytes_verified":true,
                    "encode_seconds":encode_seconds,"decode_verify_seconds":decode_verify_seconds
                }));
            }
        }
        assert!(best <= baseline_complete);
        original_total += baseline_bytes.len();
        baseline_total += baseline_complete;
        selected_total += best;
        rows.push(json!({
            "label":case["label"],"source":path,"manifest_sha256":case["sha256"],
            "original_bytes":original.len(),"baseline_existing_v1_bytes":baseline_bytes.len(),
            "baseline_mode_bytes":1,"baseline_equivalent_complete_bytes":baseline_complete,
            "framing_bytes":base.framing,"reconstruction_bytes":base.metadata.len(),
            "baseline_stream_bytes":baseline_stream_bytes,
            "training_samples":training.sample_count,"training_modelled_bits":training.modelled_bits,
            "training_direct_bits":training.direct_bits,"trained_tree_leaves":training.tree.leaves(),
            "training_seconds":training_seconds,"variants":variants,
            "selected_tool":winner,"selected_equivalent_complete_bytes":best
        }));
    }
    assert_eq!(encoded_candidates, 42);
    let mut summaries = Vec::new();
    for (i, name) in ["single_leaf", "fixed_event_classes", "learned_ma_contexts"]
        .into_iter()
        .enumerate()
    {
        for (j, rate) in [6u8, 8u8].into_iter().enumerate() {
            let total = totals[i * 2 + j];
            summaries.push(json!({"tool":name,"rate":rate,
                "equivalent_complete_bytes":total,
                "delta_vs_original_v7_candidate":total as i64-baseline_total as i64}));
        }
    }
    let receipt = json!({
        "experiment":"Learned MA tree selects online probability contexts over original-domain adaptive v7",
        "manifest":std::path::PathBuf::from(manifest_path),
        "scope":"Same seven JPEG smoke inputs; defaults and pixel path unchanged",
        "accounting":"Original candidate includes one mode byte. Every other equivalent complete cost includes unchanged v1 framing and reconstruction metadata, actual versioned descriptor/tree/parameters and all arithmetic streams.",
        "controls":["Original v7","Single-leaf adaptive calibration with 16 probability bins","Fixed namespace/event-class contexts, six leaves each with 16 bins","Learned MA contexts, at most eight leaves each with 16 bins"],
        "limitations":[
            "Original coefficient values, v7 predictions, event traversal, counters and expert-weight updates remain intact; only the probability supplied to the arithmetic coder is calibrated.",
            "Tree learning is a bounded conditional-logloss partition search with original v7 probability as an offset, not the JXL residual-histogram predictor learner.",
            "Fitted training offsets are not transmitted or used as runtime probabilities. Each leaf/probability-bin bias starts at zero and updates from decoded bits; state resets at each band.",
            "The true tree descriptor is parsed on decode; no encoder-only tree, bit labels or coefficient values are supplied to decoding.",
            "Only two online learning rates and a small tree budget were tested; this is not an optimum or a general conclusion about MA or pixel CM.",
            "SHA256 identities are copied from the existing manifest; original reconstructed bytes are compared directly.",
            "Stage timings are diagnostics, not a formal benchmark. No existing coding tool is removed."
        ],
        "encoded_candidates":encoded_candidates,"baseline_existing_v1_total_bytes":original_total,
        "baseline_equivalent_total_bytes":baseline_total,"variant_totals":summaries,
        "selected_equivalent_total_bytes":selected_total,"cases":rows
    });
    let path = std::path::PathBuf::from(receipt_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
}
