//! Opt-in seven-input experiment. No public static format is selected here.
//! The same v1 reconstruction metadata and framing are used for accounting.
use crate::{DecodeLimits, EncodeOptions, decode_jpeg, encode_jpeg, model};
use rs_jxl::jpeg::reconstruction_decode::{
    CapturedJpegCoefficients, CapturedJpegComponent, ReconstructionDecodeLimits, reconstruct_jpeg,
};
use rs_jxl::jpeg::reconstruction_metadata::ReconstructionMetadataSources;
use serde_json::json;

fn le16(d: &[u8], p: usize) -> u16 {
    u16::from_le_bytes(d[p..p + 2].try_into().unwrap())
}
fn le32(d: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(d[p..p + 4].try_into().unwrap())
}
fn le64(d: &[u8], p: usize) -> usize {
    usize::try_from(u64::from_le_bytes(d[p..p + 8].try_into().unwrap())).unwrap()
}

#[test]
#[ignore = "explicit small local manifest and receipt path required"]
fn static_probability_seven_inputs() {
    let manifest = std::env::var_os("RS_JXL_PLUS_STATIC_MANIFEST").expect("manifest");
    let out = std::env::var_os("RS_JXL_PLUS_STATIC_RECEIPT").expect("receipt destination");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert!(
        !cases.is_empty() && cases.len() <= 7,
        "small experiment only"
    );
    let options = EncodeOptions::default();
    let mut rows = Vec::new();
    for case in cases {
        let path = case["source"].as_str().unwrap();
        let original = std::fs::read(path).unwrap();
        let dynamic = encode_jpeg(&original, &options).unwrap();
        assert_eq!(
            decode_jpeg(&dynamic, &DecodeLimits::default()).unwrap(),
            original
        );
        assert_eq!(le16(&dynamic, 8), 1);
        assert_eq!(le16(&dynamic, 10), 0);
        let (width, height) = (le32(&dynamic, 12), le32(&dynamic, 16));
        let (nc, nb) = (dynamic[28] as usize, dynamic[29] as usize);
        let metadata_len = le64(&dynamic, 32);
        let mut at = 40;
        let mut components = Vec::new();
        for _ in 0..nc {
            let mut quant = [0; 64];
            for (i, q) in quant.iter_mut().enumerate() {
                *q = le16(&dynamic, at + 12 + 2 * i);
            }
            let (bw, bh) = (
                le32(&dynamic, at + 4) as usize,
                le32(&dynamic, at + 8) as usize,
            );
            components.push(model::Component {
                width: bw,
                height: bh,
                h_samp: dynamic[at] as usize,
                v_samp: dynamic[at + 1] as usize,
                quant,
                coefficients: vec![0; bw * bh * 64],
            });
            at += 140;
        }
        let lengths: Vec<_> = (0..nb).map(|i| le64(&dynamic, at + 8 * i)).collect();
        at += 8 * nb;
        let framing = at;
        let metadata = &dynamic[at..at + metadata_len];
        at += metadata_len;
        let mut streams = Vec::new();
        for length in &lengths {
            streams.push(&dynamic[at..at + length]);
            at += length;
        }
        assert_eq!(at, dynamic.len());
        model::decode_strict(&mut components, &streams).unwrap();
        let mut variants = Vec::new();
        let mut event_counts = None;
        for profile in [1, 2, 3] {
            for hash_bits in [10, 12, 14, 16] {
                let encoded = model::encode_static_profile(
                    &components,
                    options.limits.max_output_bytes,
                    hash_bits,
                    profile,
                )
                .unwrap();
                let mut decoded = components.clone();
                for c in &mut decoded {
                    c.coefficients.fill(0);
                }
                model::decode_static(
                    &mut decoded,
                    &encoded.table,
                    &encoded
                        .streams
                        .iter()
                        .map(Vec::as_slice)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                assert_eq!(decoded, components);
                let captured = CapturedJpegCoefficients {
                    width,
                    height,
                    components: decoded
                        .into_iter()
                        .map(|c| CapturedJpegComponent {
                            h_samp_factor: c.h_samp as u32,
                            v_samp_factor: c.v_samp as u32,
                            width_in_blocks: c.width as u32,
                            height_in_blocks: c.height as u32,
                            coefficients: c.coefficients,
                            quantization: c.quant.map(i32::from),
                        })
                        .collect(),
                };
                let reconstructed = reconstruct_jpeg(
                    metadata,
                    captured,
                    ReconstructionMetadataSources::default(),
                    &ReconstructionDecodeLimits::default(),
                )
                .unwrap();
                assert_eq!(reconstructed, original);
                let a = encoded.accounting;
                assert_eq!(a.table_bytes, encoded.table.len());
                assert_eq!(
                    a.stream_bytes,
                    encoded.streams.iter().map(Vec::len).sum::<usize>()
                );
                assert_eq!(a.total_bytes, a.table_bytes + a.stream_bytes);
                assert_eq!(a.profile, profile);
                let counts = (a.modelled_bits, a.direct_bits);
                if let Some(expected) = event_counts {
                    assert_eq!(
                        counts, expected,
                        "static profiles retain the same decisions"
                    );
                }
                event_counts = Some(counts);
                // The table is self describing. This accounts for its actual wire
                // representation plus unchanged framing; it is not a public format.
                let equivalent_total = framing + metadata_len + a.total_bytes;
                eprintln!(
                    "static {} profile={} hb={} dynamic={} static_equivalent={} table={} streams={}",
                    case["label"],
                    profile,
                    hash_bits,
                    dynamic.len(),
                    equivalent_total,
                    a.table_bytes,
                    a.stream_bytes
                );
                variants.push(json!({
                "profile":profile,
                "hash_bits":hash_bits, "table_entries":a.table_entries,
                "observed_buckets":a.observed_buckets, "nondefault_buckets":a.nondefault_buckets,
                "modelled_bits":a.modelled_bits, "direct_bits":a.direct_bits,
                "table_bytes":a.table_bytes, "stream_bytes":a.stream_bytes,
                "model_bytes":a.total_bytes, "equivalent_complete_bytes":equivalent_total,
                "original_bytes_verified":true,
            }));
            }
        }
        rows.push(json!({
            "label":case["label"], "source_sha256":case["sha256"],
            "original_bytes":original.len(), "dynamic_complete_bytes":dynamic.len(),
            "dynamic_stream_bytes":lengths.iter().sum::<usize>(), "bands":nb,
            "common_framing_bytes":framing, "common_reconstruction_bytes":metadata_len,
            "static":variants,
        }));
    }
    let receipt = json!({
        "experiment":"PJA v7 finite hashed-context static probabilities, two passes",
        "scope":"same seven prior smoke inputs; no full corpus",
        "profiles":{"1":"full context tuple", "2":"primary counter", "3":"mixer selector"},
        "table_cost":"actual serialized table including descriptor; all variants decode original JPEG bytes",
        "complete_bytes":"model bytes plus unchanged experimental v1 framing and reconstruction metadata; not a public static file syntax",
        "limits":"hash projection and finite table family are experimental; not a bound on static models; no speed claim",
        "cases":rows,
    });
    std::fs::write(out, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
}
