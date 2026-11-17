//! Test-only composition of JXL's reversible CfL and the unchanged v7 coder.
//! This is an internal tool experiment, not a public syntax or a replacement
//! for any existing PJA/JXL tool. CfL residuals form a new coefficient domain;
//! v7's boundary estimates are then residual-domain estimates, not physical
//! original-DCT boundary estimates. No original luma is used by the inverse.
use crate::{
    DecodeLimits, EncodeOptions, Error, Result,
    arith::{Decoder, Encoder},
    decode_jpeg, encode_jpeg, model, raw_model,
};
use rs_jxl::jpeg::{
    cfl::{self, CflFitPolicy, CflMode, JpegCfl},
    coefficient_reader::{CoefficientReadLimits, read_coefficients_with_limits},
    reader::ReadLimits,
    reconstruction_decode::{CapturedJpegCoefficients, CapturedJpegComponent, reconstruct_jpeg},
    reconstruction_metadata::ReconstructionMetadataSources,
    transcode_layout::{CFL_FIXED_POINT_RATIO_MAX, TranscodeLayout},
};
use serde_json::json;
const TILE: usize = 8;
// Equivalent experimental descriptor: mode1, profile1, JXL->JPEG permutation
// (three bytes), tile log2=3, u32 map width/height/coded length, coded maps.
// Baseline explicitly costs its own one-byte mode0 descriptor as well.
const DESCRIPTOR_HEADER: usize = 18;
fn invalid() -> Error {
    Error::Invalid("CfL-v7 experimental descriptor")
}
struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).ok_or_else(invalid)?;
        let out = self.d.get(self.p..end).ok_or_else(invalid)?;
        self.p = end;
        Ok(out)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn len(&mut self) -> Result<usize> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            .map_err(|_| invalid())
    }
}
pub(super) struct Baseline {
    pub(super) components: Vec<model::Component>,
    pub(super) metadata: Vec<u8>,
    pub(super) streams: Vec<Vec<u8>>,
    pub(super) framing: usize,
    pub(super) width: u32,
    pub(super) height: u32,
}
pub(super) fn baseline(bytes: &[u8]) -> Result<Baseline> {
    let mut r = Reader { d: bytes, p: 0 };
    if r.take(8)? != crate::MAGIC || r.u16()? != 1 || r.u16()? != 0 {
        return Err(invalid());
    }
    let (width, height) = (r.u32()?, r.u32()?);
    let _original_len = r.len()?;
    let n = r.byte()? as usize;
    let bands = r.byte()? as usize;
    if !matches!(n, 1 | 3) || !(1..=16).contains(&bands) || r.u16()? != 0 {
        return Err(invalid());
    }
    let metadata_len = r.len()?;
    let mut components = Vec::new();
    for _ in 0..n {
        let h_samp = r.byte()? as usize;
        let v_samp = r.byte()? as usize;
        if r.u16()? != 0 {
            return Err(invalid());
        }
        let (w, h) = (r.u32()? as usize, r.u32()? as usize);
        let mut quant = [0; 64];
        for q in &mut quant {
            *q = r.u16()?;
        }
        let count = w
            .checked_mul(h)
            .and_then(|n| n.checked_mul(64))
            .ok_or_else(invalid)?;
        components.push(model::Component {
            width: w,
            height: h,
            h_samp,
            v_samp,
            quant,
            coefficients: vec![0; count],
        });
    }
    let mut lens = Vec::new();
    for _ in 0..bands {
        lens.push(r.len()?);
    }
    let framing = r.p;
    let metadata = r.take(metadata_len)?.to_vec();
    let mut streams = Vec::new();
    for len in lens {
        streams.push(r.take(len)?.to_vec());
    }
    if r.p != bytes.len() {
        return Err(invalid());
    }
    model::decode_strict(
        &mut components,
        &streams.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )?;
    Ok(Baseline {
        components,
        metadata,
        streams,
        framing,
        width,
        height,
    })
}
fn transpose(i: usize) -> usize {
    (i % 8) * 8 + i / 8
}
fn checked_natural(transposed: &[i32; 64]) -> Result<[i16; 64]> {
    let mut out = [0; 64];
    for (i, v) in out.iter_mut().enumerate() {
        *v = i16::try_from(transposed[transpose(i)])
            .map_err(|_| Error::Unsupported("CfL residual exceeds v7 i16"))?;
        if *v == i16::MIN {
            return Err(Error::Unsupported("CfL residual -32768 outside v7"));
        }
    }
    Ok(out)
}
fn ratios(luma: &model::Component, target: &model::Component) -> Result<[i32; 64]> {
    let mut out = [0; 64];
    for (i, r) in out.iter_mut().enumerate() {
        if target.quant[i] == 0 {
            return Err(invalid());
        }
        *r = (2048 * i32::from(luma.quant[i])) / i32::from(target.quant[i]);
        if !(0..=CFL_FIXED_POINT_RATIO_MAX).contains(r) {
            return Err(Error::Unsupported("CfL quantization ratio"));
        }
    }
    Ok(out)
}
pub(super) struct Maps {
    order: [usize; 3],
    width: usize,
    height: usize,
    values: Vec<u8>,
}
fn validate_maps(components: &[model::Component], maps: &Maps) -> Result<()> {
    if components.len() != 3
        || maps.order.iter().any(|&i| i >= 3)
        || maps.order[0] == maps.order[1]
        || maps.order[0] == maps.order[2]
        || maps.order[1] == maps.order[2]
    {
        return Err(invalid());
    }
    let l = &components[maps.order[1]];
    if components.iter().any(|c| {
        c.width != l.width || c.height != l.height || c.h_samp != l.h_samp || c.v_samp != l.v_samp
    }) || maps.width != l.width.div_ceil(TILE)
        || maps.height != l.height.div_ceil(TILE)
        || maps
            .width
            .checked_mul(maps.height)
            .and_then(|n| n.checked_mul(2))
            != Some(maps.values.len())
    {
        return Err(invalid());
    }
    Ok(())
}
pub(super) fn encode_maps(maps: &Maps, cap: usize) -> Result<Vec<u8>> {
    let mut raw = maps.values.clone();
    let mut cd = Encoder::new(cap);
    let raw_cap = raw.len().min(cap);
    raw_model::code(&mut cd, &mut raw, raw_cap)?;
    let coded = cd.finish()?;
    if coded.len() > cap.saturating_sub(DESCRIPTOR_HEADER) || cap < DESCRIPTOR_HEADER {
        return Err(Error::Limit("CfL map descriptor bytes"));
    }
    let mut out = vec![
        1,
        1,
        maps.order[0] as u8,
        maps.order[1] as u8,
        maps.order[2] as u8,
        3,
    ];
    for v in [maps.width, maps.height, coded.len()] {
        out.extend(u32::try_from(v).map_err(|_| invalid())?.to_le_bytes());
    }
    assert_eq!(out.len(), DESCRIPTOR_HEADER);
    out.extend(coded);
    Ok(out)
}
pub(super) fn decode_maps(descriptor: &[u8], components: &[model::Component]) -> Result<Maps> {
    let mut r = Reader {
        d: descriptor,
        p: 0,
    };
    if r.byte()? != 1 || r.byte()? != 1 {
        return Err(invalid());
    }
    let order = [r.byte()? as usize, r.byte()? as usize, r.byte()? as usize];
    if r.byte()? != 3 {
        return Err(invalid());
    }
    let width = r.u32()? as usize;
    let height = r.u32()? as usize;
    let len = r.u32()? as usize;
    let n = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(2))
        .filter(|&n| n <= 1 << 20)
        .ok_or_else(invalid)?;
    let encoded = r.take(len)?;
    if r.p != descriptor.len() {
        return Err(invalid());
    }
    let mut cd = Decoder::new(encoded);
    let mut values = Vec::new();
    raw_model::code(&mut cd, &mut values, n)?;
    cd.finish_check()?;
    let maps = Maps {
        order,
        width,
        height,
        values,
    };
    validate_maps(components, &maps)?;
    Ok(maps)
}
pub(super) fn inverse(components: &mut [model::Component], maps: &Maps) -> Result<()> {
    validate_maps(components, maps)?;
    let luma = maps.order[1];
    let n = maps.width * maps.height;
    for (channel, mapoff) in [(0, 0), (2, n)] {
        let target = maps.order[channel];
        let q = ratios(&components[luma], &components[target])?;
        let (w, h) = (components[luma].width, components[luma].height);
        for by in 0..h {
            for bx in 0..w {
                let offset = (by * w + bx) * 64;
                let lum: [i16; 64] = components[luma].coefficients[offset..offset + 64]
                    .try_into()
                    .unwrap();
                let natural = &components[target].coefficients[offset..offset + 64];
                let transposed = std::array::from_fn(|i| i32::from(natural[transpose(i)]));
                let factor = maps.values[mapoff + (by / TILE) * maps.width + bx / TILE] as i8;
                let mut original = [0; 64];
                cfl::inverse_ac_transposed(&transposed, &lum, &q, factor, &mut original)
                    .map_err(|e| Error::Jpeg(e.to_string()))?;
                components[target].coefficients[offset..offset + 64].copy_from_slice(&original);
            }
        }
    }
    Ok(())
}
pub(super) fn forward(
    jpeg: &rs_jxl::jpeg::data::JpegData,
    components: &[model::Component],
    policy: CflFitPolicy,
) -> Result<(Vec<model::Component>, Maps)> {
    let layout = TranscodeLayout::new(jpeg).map_err(|e| Error::Jpeg(e.to_string()))?;
    if !layout.can_use_ac_cfl() {
        return Err(Error::Unsupported("CfL requires eligible relative 4:4:4"));
    }
    let cfl = JpegCfl::compute_with_policy(&layout, CflMode::Enabled, policy);
    if !cfl.is_enabled() {
        return Err(Error::Unsupported("CfL disabled by layout qualification"));
    }
    let order = layout.jpeg_order();
    let (mw, mh) = cfl.dimensions_in_tiles();
    let mut values = Vec::new();
    values.extend(cfl.ytox_map().iter().map(|&v| v as u8));
    values.extend(cfl.ytob_map().iter().map(|&v| v as u8));
    let maps = Maps {
        order,
        width: mw as usize,
        height: mh as usize,
        values,
    };
    validate_maps(components, &maps)?;
    let mut residual = components.to_vec();
    for (channel, &component) in order.iter().enumerate() {
        let (w, h) = layout.planes()[channel].dimensions_in_blocks();
        if (w as usize, h as usize) != (components[component].width, components[component].height) {
            return Err(invalid());
        }
        for by in 0..h {
            for bx in 0..w {
                let mut transposed = [0i32; 64];
                cfl.forward_block(channel, bx, by, &mut transposed)
                    .map_err(|e| Error::Jpeg(e.to_string()))?;
                let natural = checked_natural(&transposed)?;
                let at = (by as usize * w as usize + bx as usize) * 64;
                residual[component].coefficients[at..at + 64].copy_from_slice(&natural);
            }
        }
    }
    Ok((residual, maps))
}
pub(super) fn reconstruct(
    base: &Baseline,
    components: Vec<model::Component>,
    limits: &DecodeLimits,
) -> Result<Vec<u8>> {
    let captured = CapturedJpegCoefficients {
        width: base.width,
        height: base.height,
        components: components
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
    reconstruct_jpeg(
        &base.metadata,
        captured,
        ReconstructionMetadataSources::default(),
        &crate::container::reconstruction_limits(limits),
    )
    .map_err(|e| Error::Jpeg(e.to_string()))
}

#[test]
fn zero_cfl_factor_is_identity_before_v7() {
    let original: Vec<_> = (0..3)
        .map(|c| model::Component {
            width: 2,
            height: 2,
            h_samp: 1,
            v_samp: 1,
            quant: std::array::from_fn(|i| (1 + (i + c) % 19) as u16),
            coefficients: (0..256)
                .map(|i| ((i * 37 + c * 71) % 999) as i16 - 499)
                .collect(),
        })
        .collect();
    let baseline = model::encode(&original, 1 << 20).unwrap();
    for order in [[1, 0, 2], [0, 1, 2]] {
        let maps = Maps {
            order,
            width: 1,
            height: 1,
            values: vec![0; 2],
        };
        let mut residual = original.clone();
        for (channel, &target) in order.iter().enumerate() {
            for block in 0..4 {
                let at = block * 64;
                let input: &[i16; 64] = original[target].coefficients[at..at + 64]
                    .try_into()
                    .unwrap();
                let lum: &[i16; 64] = original[order[1]].coefficients[at..at + 64]
                    .try_into()
                    .unwrap();
                let q = ratios(&original[order[1]], &original[target]).unwrap();
                let mut transposed = [0; 64];
                if channel == 1 {
                    for i in 0..64 {
                        transposed[transpose(i)] = i32::from(input[i]);
                    }
                } else {
                    cfl::forward_ac_transposed(input, lum, &q, 0, &mut transposed).unwrap();
                }
                let natural = checked_natural(&transposed).unwrap();
                residual[target].coefficients[at..at + 64].copy_from_slice(&natural);
            }
        }
        assert_eq!(residual, original);
        let streams = model::encode(&residual, 1 << 20).unwrap();
        assert_eq!(streams, baseline);
        for c in &mut residual {
            c.coefficients.fill(0);
        }
        model::decode_strict(
            &mut residual,
            &streams.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )
        .unwrap();
        let descriptor = encode_maps(&maps, 1 << 20).unwrap();
        let decoded_maps = decode_maps(&descriptor, &residual).unwrap();
        inverse(&mut residual, &decoded_maps).unwrap();
        assert_eq!(residual, original);
        let mut bad = descriptor;
        bad.push(0);
        assert!(decode_maps(&bad, &residual).is_err());
    }
    let mut wide = [0; 64];
    wide[1] = 32768;
    assert!(checked_natural(&wide).is_err());
    wide[1] = -32768;
    assert!(checked_natural(&wide).is_err());
}

#[test]
#[ignore = "requires old seven-input smoke manifest and explicit receipt destination"]
fn cfl_then_v7_seven_inputs() {
    let manifest_path =
        std::env::var_os("RS_JXL_PLUS_TOOL_MANIFEST").expect("RS_JXL_PLUS_TOOL_MANIFEST");
    let receipt_path =
        std::env::var_os("RS_JXL_PLUS_TOOL_RECEIPT").expect("RS_JXL_PLUS_TOOL_RECEIPT");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7, "only the existing seven smoke inputs");
    let options = EncodeOptions::default();
    let mut rows = Vec::new();
    let (mut baseline_total, mut selected_total) = (0usize, 0usize);
    let (mut eligible, mut encoded_candidates) = (0usize, 0usize);
    for case in cases {
        let path = case["source"].as_str().unwrap();
        let original = std::fs::read(path).unwrap();
        assert_eq!(Some(original.len() as u64), case["original_bytes"].as_u64());
        let encoded = encode_jpeg(&original, &options).unwrap();
        assert_eq!(decode_jpeg(&encoded, &options.limits).unwrap(), original);
        let base = baseline(&encoded).unwrap();
        let dynamic_bytes = base.streams.iter().map(Vec::len).sum::<usize>();
        assert_eq!(
            base.framing + base.metadata.len() + dynamic_bytes,
            encoded.len()
        );
        let baseline_equivalent = encoded.len() + 1;
        let mut best = baseline_equivalent;
        let mut winner = "original_v7".to_owned();
        let mut variants = Vec::new();
        let read_limits = CoefficientReadLimits {
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
        };
        let jpeg = read_coefficients_with_limits(&original, &read_limits).unwrap();
        let supported = TranscodeLayout::new(&jpeg).is_ok_and(|l| l.can_use_ac_cfl());
        if supported {
            eligible += 1;
        }
        for (policy, name) in [
            (CflFitPolicy::Source, "cfl_source_then_v7"),
            (CflFitPolicy::ActualMaximum, "cfl_actual_maximum_then_v7"),
        ] {
            if !supported {
                variants.push(json!({"tool":name,"status":"not_applicable","reason":"JXL CfL requires an eligible three-component relative-4:4:4 layout"}));
                continue;
            }
            let (residual, maps) = match forward(&jpeg, &base.components, policy) {
                Ok(v) => v,
                Err(e) => {
                    variants.push(json!({"tool":name,"status":"rejected","reason":e.to_string()}));
                    continue;
                }
            };
            let descriptor = encode_maps(&maps, options.limits.max_output_bytes).unwrap();
            let streams = match model::encode(
                &residual,
                options.limits.max_output_bytes - descriptor.len(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    variants.push(json!({"tool":name,"status":"rejected","reason":e.to_string()}));
                    continue;
                }
            };
            assert_eq!(streams.len(), base.streams.len());
            let mut decoded = residual;
            for c in &mut decoded {
                c.coefficients.fill(0);
            }
            model::decode_strict(
                &mut decoded,
                &streams.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            )
            .unwrap();
            let maps_decoded = decode_maps(&descriptor, &decoded).unwrap();
            inverse(&mut decoded, &maps_decoded).unwrap();
            assert_eq!(
                decoded, base.components,
                "{name}: inverse uses decoded luma/maps only"
            );
            assert_eq!(
                reconstruct(&base, decoded, &options.limits).unwrap(),
                original,
                "{name}: original JPEG bytes"
            );
            let stream_bytes = streams.iter().map(Vec::len).sum::<usize>();
            let complete = base.framing + base.metadata.len() + descriptor.len() + stream_bytes;
            if complete < best {
                best = complete;
                winner = name.to_owned();
            }
            encoded_candidates += 1;
            eprintln!(
                "CfL->v7 {} {} baseline={} complete={} map={} streams={}",
                case["label"],
                name,
                baseline_equivalent,
                complete,
                descriptor.len() - DESCRIPTOR_HEADER,
                stream_bytes
            );
            variants.push(json!({"tool":name,"status":"encoded","coefficient_domain":"JXL CfL residuals converted back to natural order; unchanged v7 predictions run in this residual domain","maps_raw_bytes":maps.values.len(),"maps_coded_bytes":descriptor.len()-DESCRIPTOR_HEADER,"descriptor_header_bytes":DESCRIPTOR_HEADER,"descriptor_total_bytes":descriptor.len(),"coefficient_stream_bytes":stream_bytes,"bands":streams.len(),"equivalent_complete_bytes":complete,"delta_vs_original_v7_candidate":complete as i64-baseline_equivalent as i64,"decoded_coefficients_verified":true,"original_jpeg_bytes_verified":true}));
        }
        assert!(best <= baseline_equivalent);
        baseline_total += baseline_equivalent;
        selected_total += best;
        rows.push(json!({"label":case["label"],"source":path,"manifest_sha256":case["sha256"],"original_bytes":original.len(),"cfl_eligible":supported,"baseline_existing_v1_bytes":encoded.len(),"baseline_candidate_mode_bytes":1,"baseline_equivalent_complete_bytes":baseline_equivalent,"framing_bytes":base.framing,"reconstruction_bytes":base.metadata.len(),"baseline_v7_stream_bytes":dynamic_bytes,"variants":variants,"selected_tool":winner,"selected_equivalent_complete_bytes":best}));
    }
    assert!(eligible > 0 && encoded_candidates > 0);
    let receipt = json!({"experiment":"JXL reversible CfL followed by native PJA v7 coefficient coding","scope":"existing seven smoke inputs only; defaults and all existing tools retained","manifest":std::path::PathBuf::from(manifest_path),"accounting":"Equivalent complete size = unchanged v1 framing + same compressed reconstruction metadata + actual mode/descriptor/map bytes + arithmetic coefficient streams. Baseline includes an explicit one-byte mode0 descriptor. Not a public format.","limitations":["Only currently eligible 4:4:4 inputs; no subsampling extension was invented.","i32 residuals must fit v7 signed i16 excluding -32768; otherwise that candidate is rejected, never clipped.","Fitted maps are transmitted through raw_model with canonical termination; the inverse derives ratios from component quantization and uses decoded luma only.","Original PJA boundary predictors operate in the CfL residual domain; this is an additional reversible composition candidate, not equivalent physical-boundary modeling.","Manifest SHA256 is copied for reference; this test verifies full reconstructed bytes, not an independently recomputed SHA256.","No compression result removes Full/Fast/Blend or any JXL tool; selection uses complete candidate cost."],"eligible_inputs":eligible,"encoded_candidates":encoded_candidates,"baseline_equivalent_total_bytes":baseline_total,"selected_equivalent_total_bytes":selected_total,"cases":rows});
    let receipt_path = std::path::PathBuf::from(receipt_path);
    if let Some(parent) = receipt_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(receipt_path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
}
