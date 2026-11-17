//! Bounded tool-combination search on the existing seven JPEG smoke inputs.
//! This is a test representation, not the final public extension bitstream.
use crate::{
    DecodeLimits, EncodeOptions, Error, Result, decode_jpeg, encode_jpeg,
    model::{self, band_tools, context_refine},
    tool_tests::{self, Baseline, baseline, reconstruct},
};
use rs_jxl::jpeg::{
    cfl::CflFitPolicy,
    coefficient_reader::{CoefficientReadLimits, read_coefficients_with_limits},
    reader::ReadLimits,
    transcode_layout::TranscodeLayout,
};
use serde_json::{Value, json};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Coding {
    V7,
    Full,
    Fast,
    Refined(u8),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Recipe {
    coding: Coding,
    bands: usize,
}
impl Recipe {
    fn label(self) -> String {
        format!("{:?}_bands{}", self.coding, self.bands)
    }
}
struct Payload {
    descriptor: Vec<u8>,
    streams: Vec<Vec<u8>>,
    extra_state_bytes: usize,
}
fn encode(cs: &[model::Component], r: Recipe, cap: usize) -> Result<Payload> {
    if let Coding::Refined(profile) = r.coding {
        let e = context_refine::encode_refined(cs, profile, r.bands, cap)?;
        assert_eq!(e.accounting.profile, profile);
        assert_eq!(e.accounting.bands, r.bands);
        assert_eq!(e.accounting.descriptor_bytes, e.descriptor.len());
        assert_eq!(
            e.accounting.stream_bytes,
            e.streams.iter().map(Vec::len).sum::<usize>()
        );
        assert_eq!(
            e.accounting.total_bytes,
            e.accounting.stream_bytes + e.descriptor.len()
        );
        assert!(e.accounting.modelled_bits > 0);
        let _direct_events = e.accounting.direct_bits;
        Ok(Payload {
            descriptor: e.descriptor,
            streams: e.streams,
            extra_state_bytes: e.accounting.states_bytes,
        })
    } else {
        let tool = match r.coding {
            Coding::V7 => band_tools::Tool::V7,
            Coding::Full => band_tools::Tool::Full,
            Coding::Fast => band_tools::Tool::Fast,
            Coding::Refined(_) => unreachable!(),
        };
        let e = band_tools::encode_candidate(cs, tool, r.bands, cap)?;
        assert_eq!(e.accounting.bands, r.bands);
        assert_eq!(e.accounting.tool, e.descriptor[2]);
        assert_eq!(e.accounting.descriptor_bytes, e.descriptor.len());
        assert_eq!(
            e.accounting.stream_bytes,
            e.streams.iter().map(Vec::len).sum::<usize>()
        );
        assert_eq!(
            e.accounting.total_bytes,
            e.accounting.stream_bytes + e.descriptor.len()
        );
        Ok(Payload {
            descriptor: e.descriptor,
            streams: e.streams,
            extra_state_bytes: 0,
        })
    }
}
fn invalid() -> Error {
    Error::Invalid("tool-composition descriptor")
}
fn compose(transform: &[u8], inner: Payload) -> Result<Payload> {
    let mut descriptor = vec![6, 1];
    for n in [transform.len(), inner.descriptor.len()] {
        descriptor.extend(u32::try_from(n).map_err(|_| invalid())?.to_le_bytes());
    }
    descriptor.extend(transform);
    descriptor.extend(inner.descriptor);
    Ok(Payload {
        descriptor,
        streams: inner.streams,
        extra_state_bytes: inner.extra_state_bytes,
    })
}
fn split_composition(d: &[u8]) -> Result<(&[u8], &[u8])> {
    if d.len() < 10 || d[..2] != [6, 1] {
        return Err(invalid());
    }
    let a = u32::from_le_bytes(d[2..6].try_into().unwrap()) as usize;
    let b = u32::from_le_bytes(d[6..10].try_into().unwrap()) as usize;
    let mid = 10usize.checked_add(a).ok_or_else(invalid)?;
    let end = mid.checked_add(b).ok_or_else(invalid)?;
    if a == 0 || b == 0 || end != d.len() {
        return Err(invalid());
    }
    Ok((&d[10..mid], &d[mid..end]))
}
fn decode_inner(cs: &mut [model::Component], descriptor: &[u8], streams: &[&[u8]]) -> Result<()> {
    match descriptor.first() {
        Some(4) => band_tools::decode_candidate(cs, descriptor, streams),
        Some(5) => context_refine::decode_refined(cs, descriptor, streams),
        _ => Err(invalid()),
    }
}
fn decode(cs: &mut [model::Component], payload: &Payload) -> Result<()> {
    let streams: Vec<_> = payload.streams.iter().map(Vec::as_slice).collect();
    if payload.descriptor.first() == Some(&6) {
        let (maps, entropy) = split_composition(&payload.descriptor)?;
        decode_inner(cs, entropy, &streams)?;
        let maps = tool_tests::decode_maps(maps, cs)?;
        tool_tests::inverse(cs, &maps)
    } else {
        decode_inner(cs, &payload.descriptor, &streams)
    }
}
fn check_and_record(
    base: &Baseline,
    original: &[u8],
    limits: &DecodeLimits,
    payload: Payload,
    label: &str,
    stage: &str,
    reference: usize,
    encode_seconds: f64,
) -> (usize, Value) {
    let mut cs = base.components.clone();
    for c in &mut cs {
        c.coefficients.fill(0);
    }
    let t = Instant::now();
    decode(&mut cs, &payload).unwrap();
    assert_eq!(cs, base.components, "{label}: coefficients");
    assert_eq!(
        reconstruct(base, cs, limits).unwrap(),
        original,
        "{label}: original JPEG bytes"
    );
    let decode_verify_seconds = t.elapsed().as_secs_f64();
    let stream_bytes = payload.streams.iter().map(Vec::len).sum::<usize>();
    let bands = payload.streams.len();
    let framing_bytes = 40 + 140 * base.components.len() + 8 * bands;
    let complete = framing_bytes + base.metadata.len() + payload.descriptor.len() + stream_bytes;
    let row = json!({
        "tool":label,"stage":stage,"status":"encoded","bands":bands,
        "descriptor_bytes":payload.descriptor.len(),"stream_bytes":stream_bytes,
        "framing_bytes":framing_bytes,"band_length_bytes":8*bands,
        "arithmetic_termination_bytes_included_in_streams":4*bands,
        "reconstruction_bytes":base.metadata.len(),"equivalent_complete_bytes":complete,
        "delta_vs_original_v7_candidate":complete as i64-reference as i64,
        "extra_calibration_state_bytes_per_band":payload.extra_state_bytes,
        "encode_seconds":encode_seconds,"decode_verify_seconds":decode_verify_seconds,
        "original_coefficients_verified":true,"original_jpeg_bytes_verified":true
    });
    (complete, row)
}

#[test]
fn combination_descriptor_is_self_delimiting() {
    let p = compose(
        &[1, 2, 3],
        Payload {
            descriptor: vec![4, 1, 0, 1],
            streams: vec![],
            extra_state_bytes: 0,
        },
    )
    .unwrap();
    assert_eq!(
        split_composition(&p.descriptor).unwrap(),
        (&[1, 2, 3][..], &[4, 1, 0, 1][..])
    );
    for n in 0..p.descriptor.len() {
        assert!(split_composition(&p.descriptor[..n]).is_err());
    }
    let mut bad = p.descriptor;
    bad.push(0);
    assert!(split_composition(&bad).is_err());
}

#[test]
#[ignore = "requires existing seven-input manifest and an explicit search receipt destination"]
fn compression_tools_seven_inputs() {
    let manifest_path =
        std::env::var_os("RS_JXL_PLUS_SEARCH_MANIFEST").expect("RS_JXL_PLUS_SEARCH_MANIFEST");
    let receipt_path =
        std::env::var_os("RS_JXL_PLUS_SEARCH_RECEIPT").expect("RS_JXL_PLUS_SEARCH_RECEIPT");
    let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7);
    let options = EncodeOptions::default();
    let (mut baseline_total, mut selected_total) = (0usize, 0usize);
    let mut stage_totals = [0usize; 3];
    let mut checked = 0usize;
    let mut rows = Vec::new();
    for case in cases {
        let path = case["source"].as_str().unwrap();
        let original = std::fs::read(path).unwrap();
        assert_eq!(Some(original.len() as u64), case["original_bytes"].as_u64());
        let encoded = encode_jpeg(&original, &options).unwrap();
        assert_eq!(decode_jpeg(&encoded, &options.limits).unwrap(), original);
        let base = baseline(&encoded).unwrap();
        let source = base.streams.len();
        let mcu_rows = base.components[0].height / base.components[0].v_samp;
        let reference = encoded.len() + 1;
        let mut best = reference;
        let mut selected = "original_v7".to_owned();
        let mut variants = Vec::new();
        let mut bands = vec![1, 2, 4, source];
        bands.retain(|&b| b <= mcu_rows && b <= 16);
        bands.sort_unstable();
        bands.dedup();
        let mut best_v7 = (
            usize::MAX,
            Recipe {
                coding: Coding::V7,
                bands: source,
            },
        );
        let mut best_legacy = (
            usize::MAX,
            Recipe {
                coding: Coding::Full,
                bands: source,
            },
        );
        for recipe in bands
            .into_iter()
            .map(|bands| Recipe {
                coding: Coding::V7,
                bands,
            })
            .chain([
                Recipe {
                    coding: Coding::Full,
                    bands: source,
                },
                Recipe {
                    coding: Coding::Fast,
                    bands: source,
                },
            ])
        {
            let t = Instant::now();
            let payload =
                encode(&base.components, recipe, options.limits.max_output_bytes).unwrap();
            if recipe.coding == Coding::V7 && recipe.bands == source {
                assert_eq!(payload.streams, base.streams);
            }
            let label = recipe.label();
            let (size, row) = check_and_record(
                &base,
                &original,
                &options.limits,
                payload,
                &label,
                "bands_and_models",
                reference,
                t.elapsed().as_secs_f64(),
            );
            eprintln!(
                "Tools {} {label} baseline={reference} complete={size}",
                case["label"]
            );
            if recipe.coding == Coding::V7 {
                if size < best_v7.0 {
                    best_v7 = (size, recipe);
                }
            } else if size < best_legacy.0 {
                best_legacy = (size, recipe);
            }
            if size < best {
                best = size;
                selected = label;
            }
            checked += 1;
            variants.push(row);
        }
        let after_models = best;
        stage_totals[0] += best;
        let mut refine_bands = vec![source, best_v7.1.bands];
        refine_bands.sort_unstable();
        refine_bands.dedup();
        let mut best_refined = (
            usize::MAX,
            Recipe {
                coding: Coding::Refined(1),
                bands: source,
            },
        );
        for b in refine_bands {
            for profile in [1, 2, 3] {
                let recipe = Recipe {
                    coding: Coding::Refined(profile),
                    bands: b,
                };
                let t = Instant::now();
                let payload =
                    encode(&base.components, recipe, options.limits.max_output_bytes).unwrap();
                let label = recipe.label();
                let (size, row) = check_and_record(
                    &base,
                    &original,
                    &options.limits,
                    payload,
                    &label,
                    "fine_context",
                    reference,
                    t.elapsed().as_secs_f64(),
                );
                eprintln!(
                    "Tools {} {label} baseline={reference} complete={size}",
                    case["label"]
                );
                if size < best_refined.0 {
                    best_refined = (size, recipe);
                }
                if size < best {
                    best = size;
                    selected = label;
                }
                checked += 1;
                variants.push(row);
            }
        }
        let after_refine = best;
        stage_totals[1] += best;
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
        let cfl_eligible = TranscodeLayout::new(&jpeg).is_ok_and(|l| l.can_use_ac_cfl());
        if cfl_eligible {
            for (policy, name) in [
                (CflFitPolicy::Source, "cfl_source"),
                (CflFitPolicy::ActualMaximum, "cfl_actual_maximum"),
            ] {
                let t = Instant::now();
                let (residual, maps) = match tool_tests::forward(&jpeg, &base.components, policy) {
                    Ok(v) => v,
                    Err(e) => {
                        variants.push(json!({"tool":name,"stage":"cfl_combination","status":"rejected","reason":e.to_string()}));
                        continue;
                    }
                };
                let map_descriptor =
                    tool_tests::encode_maps(&maps, options.limits.max_output_bytes).unwrap();
                let transform_seconds = t.elapsed().as_secs_f64();
                for recipe in [best_v7.1, best_legacy.1, best_refined.1] {
                    let t = Instant::now();
                    let payload = encode(
                        &residual,
                        recipe,
                        options.limits.max_output_bytes - map_descriptor.len() - 10,
                    )
                    .unwrap();
                    let payload = compose(&map_descriptor, payload).unwrap();
                    let label = format!("{name}+{}", recipe.label());
                    let (size, row) = check_and_record(
                        &base,
                        &original,
                        &options.limits,
                        payload,
                        &label,
                        "cfl_combination",
                        reference,
                        t.elapsed().as_secs_f64() + transform_seconds,
                    );
                    eprintln!(
                        "Tools {} {label} baseline={reference} complete={size}",
                        case["label"]
                    );
                    if size < best {
                        best = size;
                        selected = label;
                    }
                    checked += 1;
                    variants.push(row);
                }
            }
        }
        stage_totals[2] += best;
        baseline_total += reference;
        selected_total += best;
        rows.push(json!({"label":case["label"],"source":path,"manifest_sha256":case["sha256"],"original_bytes":original.len(),
            "baseline_existing_v1_bytes":encoded.len(),"baseline_equivalent_complete_bytes":reference,"source_bands":source,
            "cfl_eligible":cfl_eligible,"best_after_models":after_models,"best_after_refine":after_refine,
            "selected_tool":selected,"selected_equivalent_complete_bytes":best,"variants":variants}));
    }
    let receipt = json!({
        "experiment":"V7 band partition / legacy Full-Fast / hierarchical probability refinement / actual CfL compositions",
        "scope":"Existing seven JPEG smoke inputs only; native Rust; all tools retained; defaults unchanged",
        "manifest":std::path::PathBuf::from(manifest_path),"verified_candidates":checked,
        "baseline_equivalent_total_bytes":baseline_total,"selected_equivalent_total_bytes":selected_total,
        "best_after_models_total_bytes":stage_totals[0],"best_after_refine_total_bytes":stage_totals[1],"best_after_cfl_total_bytes":stage_totals[2],
        "accounting":"Actual descriptors and arithmetic streams + same reconstruction metadata + 40+140*components+8*actual_bands framing. Original mode costs 1 byte. CfL composition additionally carries its real maps, inner entropy descriptor and 10-byte length-delimited composition header.",
        "limitations":["Finite search on seven fixed samples, not corpus generalization or a speed comparison.","V7 band changes alter model reset/warmup and boundary prediction, not just framing overhead.","Full/Fast retain original source band rules.","Refinement is tried at source and best raw-V7 band counts, not an exhaustive profile/band product.","CfL tests both existing fitting policies with the best raw-domain V7, legacy and refinement recipes; these are real encodes, not sums of separate gains.","CfL maps and coding descriptors are parsed during decoding; the inverse uses decoded luma only.","The experimental descriptors are not the final public Plus wire format.","Manifest hashes are references; every reconstructed original byte is compared."],
        "cases":rows
    });
    let path = std::path::PathBuf::from(receipt_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
}
