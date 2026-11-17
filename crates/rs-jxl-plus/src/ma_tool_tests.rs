//! Finite test-only JXL MA-tree / Weighted prediction -> PJA v7 experiment.
//! JPEG quantized coefficient planes are the data domain, not image pixels.
//! Trees are learned with rs-jxl's public MA learner, transmitted as bounded
//! explicit nodes, parsed with Tree::new, and used with decoded causal values.
//! This does not change defaults or remove any existing tool.
use crate::tool_tests::{baseline, reconstruct};
use crate::{
    EncodeOptions, Error, Result,
    arith::{Decoder, Encoder},
    decode_jpeg, encode_jpeg, model, raw_model,
};
use rs_jxl::modular::{
    ma_train::{self, Sample, TrainOptions},
    predictor::{Neighbors, Predictor, predict_one},
    tree::{Node, Properties, Tree},
    weighted::{self, Header},
};
use serde_json::json;
use std::time::Instant;
const SAMPLE_CAP: usize = 4096;
const LEAF_CAP: usize = 8;
const NODE_CAP: usize = 2 * LEAF_CAP - 1;
const OUTER_HEADER: usize = 6;
const PROPERTIES: [usize; 7] = [0, 4, 5, 9, 10, 11, 15];
const PREDICTORS: [Predictor; 5] = [
    Predictor::Zero,
    Predictor::Left,
    Predictor::Top,
    Predictor::Gradient,
    Predictor::Weighted,
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scope {
    Dc,
    All,
}
impl Scope {
    fn count(self) -> usize {
        if self == Self::Dc { 1 } else { 64 }
    }
    fn id(self) -> u8 {
        if self == Self::Dc { 0 } else { 1 }
    }
    fn name(self) -> &'static str {
        if self == Self::Dc {
            "ma_dc_then_v7"
        } else {
            "ma_all_frequencies_then_v7"
        }
    }
}
fn invalid() -> Error {
    Error::Invalid("MA-v7 experimental descriptor")
}
fn jxl_error(e: impl std::fmt::Display) -> Error {
    Error::Jpeg(e.to_string())
}
fn plane(c: &model::Component, freq: usize) -> Vec<i32> {
    c.coefficients
        .chunks_exact(64)
        .map(|b| i32::from(b[freq]))
        .collect()
}
fn neighborhood(values: &[i32], width: usize, x: usize, y: usize) -> Result<Neighbors> {
    Neighbors::from_plane(values, width, width, x, y).map_err(jxl_error)
}
fn wp_predict(wp: &mut weighted::State, n: Neighbors, x: usize, y: usize) -> Result<(i32, i32)> {
    wp.predict(x, y, n.top, n.left, n.topright, n.topleft, n.toptop)
        .map_err(jxl_error)
}
struct Learned {
    tree: Tree,
    header: Header,
    scope: Scope,
    samples: usize,
    population: usize,
}
fn train(components: &[model::Component], scope: Scope) -> Result<Learned> {
    // Only property 8 uses Properties::previous_gradient. Other selected
    // properties depend on current causal neighbors/WP output. If property 8
    // is added later, advance Properties on EVERY population position first.
    assert!(
        !PROPERTIES.contains(&8),
        "sampling-only Properties excludes stateful property 8"
    );
    let population = components.iter().try_fold(0usize, |sum, c| {
        sum.checked_add(
            c.width
                .checked_mul(c.height)
                .and_then(|v| v.checked_mul(scope.count()))
                .ok_or_else(invalid)?,
        )
        .ok_or_else(invalid)
    })?;
    if population == 0 {
        return Err(invalid());
    }
    let stride = population.div_ceil(SAMPLE_CAP);
    let header = Header::default();
    let mut samples = Vec::new();
    let mut global = 0usize;
    for (ci, c) in components.iter().enumerate() {
        for freq in 0..scope.count() {
            let values = plane(c, freq);
            let mut wp = weighted::State::new(header, c.width).map_err(jxl_error)?;
            let mut props = Properties::default();
            for y in 0..c.height {
                for x in 0..c.width {
                    let n = neighborhood(&values, c.width, x, y)?;
                    let (pred, error) = wp_predict(&mut wp, n, x, y)?;
                    let at = y * c.width + x;
                    let value = values[at];
                    if global.is_multiple_of(stride) {
                        let properties = props.calculate(n, x, y, ci * 64 + freq, 0, error);
                        let mut residuals = Vec::with_capacity(PREDICTORS.len());
                        for predictor in PREDICTORS {
                            let p = predict_one(predictor, n, Some(i64::from(pred)))
                                .map_err(jxl_error)?;
                            residuals
                                .push(i32::try_from(i64::from(value) - p).map_err(|_| invalid())?);
                        }
                        samples.push(Sample {
                            properties: properties.to_vec(),
                            residuals,
                            weight: 1,
                        });
                    }
                    wp.update(value, x, y).map_err(jxl_error)?;
                    global += 1;
                }
            }
        }
    }
    assert_eq!(global, population);
    assert!(!samples.is_empty() && samples.len() <= SAMPLE_CAP);
    let options = TrainOptions {
        predictors: PREDICTORS.to_vec(),
        properties: PROPERTIES.to_vec(),
        max_leaves: LEAF_CAP,
        max_property_values: 64,
        parallel: false,
        parallel_nodes: false,
        histogram_cache_bytes: 8 << 20,
        ..TrainOptions::default()
    };
    let tree = ma_train::train(&samples, &options).map_err(jxl_error)?;
    check_tree(&tree)?;
    Ok(Learned {
        tree,
        header,
        scope,
        samples: samples.len(),
        population,
    })
}
fn check_tree(tree: &Tree) -> Result<()> {
    if tree.nodes().len() > NODE_CAP || tree.leaves() > LEAF_CAP {
        return Err(invalid());
    }
    for node in tree.nodes() {
        match node {
            Node::Split { property, .. } => {
                if !PROPERTIES.contains(property) {
                    return Err(invalid());
                }
            }
            Node::Leaf { predictor, .. } => {
                if !PREDICTORS.contains(predictor) {
                    return Err(invalid());
                }
            }
        }
    }
    Ok(())
}
fn transform(
    components: &[model::Component],
    learned: &Learned,
    inverse: bool,
) -> Result<Vec<model::Component>> {
    let mut out = components.to_vec();
    for (ci, c) in components.iter().enumerate() {
        for freq in 0..learned.scope.count() {
            let input = plane(c, freq);
            let mut restored = vec![0i32; input.len()];
            let mut props = Properties::default();
            let mut wp = weighted::State::new(learned.header, c.width).map_err(jxl_error)?;
            for y in 0..c.height {
                for x in 0..c.width {
                    let at = y * c.width + x;
                    let n = neighborhood(if inverse { &restored } else { &input }, c.width, x, y)?;
                    let (wpred, error) = wp_predict(&mut wp, n, x, y)?;
                    let properties = props.calculate(n, x, y, ci * 64 + freq, 0, error);
                    let (_, predictor, offset, multiplier) =
                        learned.tree.lookup(&properties).map_err(jxl_error)?;
                    let prediction = predict_one(predictor, n, Some(i64::from(wpred)))
                        .map_err(jxl_error)?
                        + i64::from(offset);
                    let value;
                    if inverse {
                        value = prediction + i64::from(input[at]) * i64::from(multiplier);
                        let original = i16::try_from(value)
                            .map_err(|_| Error::Invalid("MA reconstructed coefficient range"))?;
                        restored[at] = i32::from(original);
                        out[ci].coefficients[at * 64 + freq] = original;
                        wp.update(i32::from(original), x, y).map_err(jxl_error)?;
                    } else {
                        value = i64::from(input[at]) - prediction;
                        if value % i64::from(multiplier) != 0 {
                            return Err(Error::Unsupported(
                                "MA leaf multiplier is not reversible over full plane",
                            ));
                        }
                        let residual = i16::try_from(value / i64::from(multiplier))
                            .map_err(|_| Error::Unsupported("MA residual exceeds v7 i16"))?;
                        if residual == i16::MIN {
                            return Err(Error::Unsupported("MA residual -32768 outside v7"));
                        }
                        out[ci].coefficients[at * 64 + freq] = residual;
                        wp.update(input[at], x, y).map_err(jxl_error)?;
                    }
                }
            }
        }
    }
    Ok(out)
}
fn put32(out: &mut Vec<u8>, value: u32) {
    out.extend(value.to_le_bytes());
}
fn serialize(learned: &Learned, cap: usize) -> Result<(Vec<u8>, usize)> {
    check_tree(&learned.tree)?;
    let mut raw = b"MAT1".to_vec();
    raw.extend([1, learned.scope.id()]);
    let h = learned.header;
    for v in [h.p1_c, h.p2_c, h.p3_ca, h.p3_cb, h.p3_cc, h.p3_cd, h.p3_ce] {
        raw.extend(v.to_le_bytes());
    }
    for v in h.w {
        put32(&mut raw, v);
    }
    put32(&mut raw, learned.tree.nodes().len() as u32);
    for node in learned.tree.nodes() {
        match *node {
            Node::Split {
                property,
                threshold,
                greater,
                less_equal,
            } => {
                raw.extend([0, property as u8]);
                raw.extend(threshold.to_le_bytes());
                put32(&mut raw, greater as u32);
                put32(&mut raw, less_equal as u32);
            }
            Node::Leaf {
                predictor,
                offset,
                multiplier,
            } => {
                raw.extend([1, predictor as u8]);
                raw.extend(offset.to_le_bytes());
                put32(&mut raw, multiplier);
            }
        }
    }
    let raw_len = raw.len();
    let mut cd = Encoder::new(cap);
    raw_model::code(&mut cd, &mut raw, 4096)?;
    let bytes = cd.finish()?;
    if cap < OUTER_HEADER || bytes.len() > cap - OUTER_HEADER {
        return Err(Error::Limit("MA descriptor bytes"));
    }
    let mut out = vec![2, 1];
    put32(&mut out, u32::try_from(bytes.len()).map_err(|_| invalid())?);
    assert_eq!(out.len(), OUTER_HEADER);
    out.extend(bytes);
    Ok((out, raw_len))
}
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(invalid)?;
        let out = self.bytes.get(self.pos..end).ok_or_else(invalid)?;
        self.pos = end;
        Ok(out)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}
fn deserialize(descriptor: &[u8]) -> Result<Learned> {
    let mut outer = Reader {
        bytes: descriptor,
        pos: 0,
    };
    if outer.byte()? != 2 || outer.byte()? != 1 {
        return Err(invalid());
    }
    let len = outer.u32()? as usize;
    let coded = outer.take(len)?;
    if outer.pos != descriptor.len() {
        return Err(invalid());
    }
    let mut cd = Decoder::new(coded);
    let mut raw = Vec::new();
    raw_model::code(&mut cd, &mut raw, 4096)?;
    cd.finish_check()?;
    let mut r = Reader {
        bytes: &raw,
        pos: 0,
    };
    if r.take(4)? != b"MAT1" || r.byte()? != 1 {
        return Err(invalid());
    }
    let scope = match r.byte()? {
        0 => Scope::Dc,
        1 => Scope::All,
        _ => return Err(invalid()),
    };
    let header = Header {
        p1_c: r.i32()?,
        p2_c: r.i32()?,
        p3_ca: r.i32()?,
        p3_cb: r.i32()?,
        p3_cc: r.i32()?,
        p3_cd: r.i32()?,
        p3_ce: r.i32()?,
        w: [r.u32()?, r.u32()?, r.u32()?, r.u32()?],
    };
    header.validate().map_err(jxl_error)?;
    let count = r.u32()? as usize;
    if count == 0 || count > NODE_CAP {
        return Err(invalid());
    }
    let mut nodes = Vec::new();
    for _ in 0..count {
        let tag = r.byte()?;
        let parameter = r.byte()?;
        let signed = r.i32()?;
        let next = r.u32()?;
        nodes.push(match tag {
            0 => Node::Split {
                property: parameter as usize,
                threshold: signed,
                greater: next as usize,
                less_equal: r.u32()? as usize,
            },
            1 => Node::Leaf {
                predictor: Predictor::try_from(u32::from(parameter)).map_err(jxl_error)?,
                offset: signed,
                multiplier: next,
            },
            _ => return Err(invalid()),
        });
    }
    if r.pos != raw.len() {
        return Err(invalid());
    }
    let tree = Tree::new(nodes).map_err(jxl_error)?;
    check_tree(&tree)?;
    Ok(Learned {
        tree,
        header,
        scope,
        samples: 0,
        population: 0,
    })
}

fn fixture() -> Vec<model::Component> {
    (0..2)
        .map(|ci| {
            let (w, h) = (32usize, 32usize);
            let mut coefficients = vec![0i16; w * h * 64];
            for y in 0..h {
                for x in 0..w {
                    let at = (y * w + x) * 64;
                    coefficients[at] = if ci == 0 {
                        (1000 + x + 3 * y) as i16
                    } else if (x + y) % 2 == 0 {
                        2000
                    } else {
                        -2000
                    };
                    for k in 1..64 {
                        coefficients[at + k] = if (x + y + k) % 13 == 0 {
                            ((x * 17 + y * 31 + k * 7) % 401) as i16 - 200
                        } else {
                            0
                        };
                    }
                }
            }
            model::Component {
                width: w,
                height: h,
                h_samp: 1,
                v_samp: 1,
                quant: [1; 64],
                coefficients,
            }
        })
        .collect()
}
#[test]
fn learned_tree_and_weighted_roundtrip_are_nontrivial() {
    let original = fixture();
    let learned = train(&original, Scope::Dc).unwrap();
    assert!(
        learned.tree.leaves() > 1,
        "the synthetic populations need distinct prediction regimes: {:?}",
        learned.tree.nodes()
    );
    assert!(
        learned
            .tree
            .predictors()
            .iter()
            .any(|&p| p != Predictor::Zero)
    );
    let residual = transform(&original, &learned, false).unwrap();
    assert_ne!(
        residual, original,
        "a learned predictor must actually transform values"
    );
    let (desc, _) = serialize(&learned, 1 << 20).unwrap();
    let decoded_tree = deserialize(&desc).unwrap();
    assert_eq!(decoded_tree.tree.tokens(), learned.tree.tokens());
    assert_eq!(decoded_tree.header, learned.header);
    let streams = model::encode(&residual, 1 << 24).unwrap();
    let mut decoded = residual;
    for c in &mut decoded {
        c.coefficients.fill(0);
    }
    model::decode_strict(
        &mut decoded,
        &streams.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(transform(&decoded, &decoded_tree, true).unwrap(), original);
    let mut bad = desc;
    bad.push(0);
    assert!(deserialize(&bad).is_err());
}
#[test]
fn zero_tree_control_preserves_full_v7_streams() {
    let mut original = fixture();
    for c in &mut original {
        c.width = 2;
        c.height = 2;
        c.coefficients.truncate(256);
    }
    let learned = Learned {
        tree: Tree::new(vec![Node::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
        }])
        .unwrap(),
        header: Header::default(),
        scope: Scope::All,
        samples: 0,
        population: 0,
    };
    let residual = transform(&original, &learned, false).unwrap();
    assert_eq!(residual, original);
    assert_eq!(
        model::encode(&residual, 1 << 20).unwrap(),
        model::encode(&original, 1 << 20).unwrap()
    );
    let (desc, _) = serialize(&learned, 1 << 20).unwrap();
    assert_eq!(
        transform(&residual, &deserialize(&desc).unwrap(), true).unwrap(),
        original
    );
}
#[test]
fn weighted_leaf_is_decoded_and_never_clips_residuals() {
    let original = fixture();
    let learned = Learned {
        tree: Tree::new(vec![Node::Leaf {
            predictor: Predictor::Weighted,
            offset: 0,
            multiplier: 1,
        }])
        .unwrap(),
        header: Header::default(),
        scope: Scope::Dc,
        samples: 0,
        population: 0,
    };
    let residual = transform(&original, &learned, false).unwrap();
    let (desc, _) = serialize(&learned, 1 << 20).unwrap();
    let parsed = deserialize(&desc).unwrap();
    assert_eq!(transform(&residual, &parsed, true).unwrap(), original);
    let oversized = Learned {
        tree: Tree::new(vec![Node::Leaf {
            predictor: Predictor::Zero,
            offset: i32::MAX,
            multiplier: 1,
        }])
        .unwrap(),
        ..learned
    };
    assert!(transform(&original, &oversized, false).is_err());
}

#[test]
#[ignore = "requires old seven-input smoke manifest and explicit MA receipt destination"]
fn ma_tree_then_v7_seven_inputs() {
    let manifest_path =
        std::env::var_os("RS_JXL_PLUS_MA_MANIFEST").expect("RS_JXL_PLUS_MA_MANIFEST");
    let receipt_path = std::env::var_os("RS_JXL_PLUS_MA_RECEIPT").expect("RS_JXL_PLUS_MA_RECEIPT");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7, "only the existing seven smoke inputs");
    let options = EncodeOptions::default();
    let mut rows = Vec::new();
    let (mut baseline_total, mut selected_total, mut encoded_candidates) = (0usize, 0usize, 0usize);
    for case in cases {
        let original = std::fs::read(case["source"].as_str().unwrap()).unwrap();
        assert_eq!(Some(original.len() as u64), case["original_bytes"].as_u64());
        let encoded = encode_jpeg(&original, &options).unwrap();
        assert_eq!(decode_jpeg(&encoded, &options.limits).unwrap(), original);
        let base = baseline(&encoded).unwrap();
        let baseline_equivalent = encoded.len() + 1;
        let mut best = baseline_equivalent;
        let mut selected = "original_v7";
        let mut variants = Vec::new();
        for scope in [Scope::Dc, Scope::All] {
            let start = Instant::now();
            let learned = match train(&base.components, scope) {
                Ok(v) => v,
                Err(e) => {
                    variants.push(json!({"tool":scope.name(),"status":"rejected","stage":"tree_learning","reason":e.to_string()}));
                    continue;
                }
            };
            let train_seconds = start.elapsed().as_secs_f64();
            let t = Instant::now();
            let residual = match transform(&base.components, &learned, false) {
                Ok(v) => v,
                Err(e) => {
                    variants.push(json!({"tool":scope.name(),"status":"rejected","stage":"coefficient_transform","reason":e.to_string(),"train_seconds":train_seconds}));
                    continue;
                }
            };
            let forward_seconds = t.elapsed().as_secs_f64();
            let (descriptor, raw_tree_bytes) =
                serialize(&learned, options.limits.max_output_bytes).unwrap();
            let t = Instant::now();
            let streams = match model::encode(
                &residual,
                options.limits.max_output_bytes - descriptor.len(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    variants.push(json!({"tool":scope.name(),"status":"rejected","stage":"v7_coding","reason":e.to_string()}));
                    continue;
                }
            };
            let v7_encode_seconds = t.elapsed().as_secs_f64();
            assert_eq!(streams.len(), base.streams.len());
            let t = Instant::now();
            let mut decoded = residual;
            for c in &mut decoded {
                c.coefficients.fill(0);
            }
            model::decode_strict(
                &mut decoded,
                &streams.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            )
            .unwrap();
            let decoded_tree = deserialize(&descriptor).unwrap();
            let restored = transform(&decoded, &decoded_tree, true).unwrap();
            assert_eq!(
                restored,
                base.components,
                "{} original coefficients",
                scope.name()
            );
            assert_eq!(
                reconstruct(&base, restored, &options.limits).unwrap(),
                original,
                "{} original JPEG bytes",
                scope.name()
            );
            let decode_inverse_verify_seconds = t.elapsed().as_secs_f64();
            let stream_bytes = streams.iter().map(Vec::len).sum::<usize>();
            let complete = base.framing + base.metadata.len() + descriptor.len() + stream_bytes;
            if complete < best {
                best = complete;
                selected = scope.name();
            }
            encoded_candidates += 1;
            let weighted_leaves = learned
                .tree
                .nodes()
                .iter()
                .filter(|n| {
                    matches!(
                        n,
                        Node::Leaf {
                            predictor: Predictor::Weighted,
                            ..
                        }
                    )
                })
                .count();
            eprintln!(
                "MA->v7 {} {} baseline={} total={} tree={} streams={} leaves={} samples={} train={:.3}s",
                case["label"],
                scope.name(),
                baseline_equivalent,
                complete,
                descriptor.len(),
                stream_bytes,
                learned.tree.leaves(),
                learned.samples,
                train_seconds
            );
            variants.push(json!({"tool":scope.name(),"status":"encoded","coefficient_domain":"per-frequency MA residuals, unchanged v7 model then operates in this residual domain","tree_nodes":learned.tree.nodes().len(),"tree_leaves":learned.tree.leaves(),"weighted_leaves":weighted_leaves,"uses_weighted_or_property15":learned.tree.uses_weighted(),"training_population":learned.population,"training_samples":learned.samples,"tree_raw_bytes":raw_tree_bytes,"tree_coded_bytes":descriptor.len()-OUTER_HEADER,"descriptor_header_bytes":OUTER_HEADER,"descriptor_total_bytes":descriptor.len(),"coefficient_stream_bytes":stream_bytes,"equivalent_complete_bytes":complete,"delta_vs_original_v7_candidate":complete as i64-baseline_equivalent as i64,"decoded_tree_verified":true,"original_coefficients_verified":true,"original_jpeg_bytes_verified":true,"train_seconds":train_seconds,"forward_seconds":forward_seconds,"v7_encode_seconds":v7_encode_seconds,"decode_inverse_verify_seconds":decode_inverse_verify_seconds}));
        }
        assert!(best <= baseline_equivalent);
        baseline_total += baseline_equivalent;
        selected_total += best;
        rows.push(json!({"label":case["label"],"source":case["source"],"manifest_sha256":case["sha256"],"original_bytes":original.len(),"baseline_existing_v1_bytes":encoded.len(),"baseline_candidate_mode_bytes":1,"baseline_equivalent_complete_bytes":baseline_equivalent,"framing_bytes":base.framing,"reconstruction_bytes":base.metadata.len(),"variants":variants,"selected_tool":selected,"selected_equivalent_complete_bytes":best}));
    }
    assert!(encoded_candidates > 0);
    let receipt = json!({"experiment":"JXL learned MA tree and Weighted coefficient prediction followed by native PJA v7","scope":"existing seven JPEG smoke inputs only; not pixel-lossless image conclusions","manifest":std::path::PathBuf::from(manifest_path),"predictors":["Zero","Left","Top","Gradient","Weighted"],"properties":PROPERTIES,"max_leaves":LEAF_CAP,"max_training_samples":SAMPLE_CAP,"max_property_values":64,"sampling":"Full population deterministic spacing. Every coefficient advances WP predict/update; only sampled positions build properties/residual arrays. Stateful property8 is excluded and asserted.","weighted_header":"All default Header values are explicitly serialized, parsed and validated. State resets at each original-frequency plane origin.","accounting":"Equivalent complete bytes = unchanged v1 framing + same compressed reconstruction metadata + six-byte mode/version/length descriptor + actual raw_model-coded tree/WP header + coefficient streams; original-v7 candidate costs one explicit mode0 byte.","limitations":["Two bounded learned-tree candidates (DC-only/all64); no exhaustive predictor/tree/transform search.","Tree payload is an actual bounded experimental node encoding rebuilt with Tree::new, not the standard-JXL MA wire representation.","Training includes Weighted; probability coding remains original adaptive PJA v7. This is not the static-probability experiment.","v7 boundary estimates operate in the externally transformed residual domain. This is an extra tool combination, not equivalent source-JXL DC coding.","Original-frequency MA causal dependencies can cross PJA band boundaries; all band streams are decoded before inverse MA.","Nondivisible multipliers or residuals outside supported v7 i16 are rejected, never clamped.","Manifest SHA256 is copied for reference; full reconstructed original bytes are compared.","Timing is diagnostic and includes different stages, not a formal performance benchmark.","No existing PJA Full/Fast/Blend or JXL tool is removed, and public defaults are unchanged."],"encoded_candidates":encoded_candidates,"baseline_equivalent_total_bytes":baseline_total,"selected_equivalent_total_bytes":selected_total,"cases":rows});
    let path = std::path::PathBuf::from(receipt_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
}
