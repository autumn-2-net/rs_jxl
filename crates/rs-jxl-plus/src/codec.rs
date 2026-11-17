//! Typed composition of native coefficient/sample coding tools. This is not
//! an on-disk format and contains no PJA archive or stream-version protocol.
//! Algorithm selection derives from PJA pjacodec.pas, MIT, revision
//! 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; LICENSE-PJA/NOTICE-PJA.
use crate::{
    DecodeLimits, Error, Result,
    arith::{Decoder, Encoder},
    jls, jpeg_native as jpeg, model, raw_model, sample_model,
};
mod reconstruction;
use reconstruction::*;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Model {
    Full,
    Fast,
    V7,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SampleMode {
    Med,
    Blend,
    Symbols,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    JpegDct(Model),
    JpegLossless(SampleMode),
    JpegLs(SampleMode),
}
#[derive(Clone, Debug)]
pub(crate) struct Options {
    pub limits: DecodeLimits,
    pub model: Model,
    pub blend: bool,
    pub verify: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            limits: DecodeLimits::default(),
            model: Model::V7,
            blend: true,
            verify: true,
        }
    }
}

/// JPEG reconstruction decisions that cannot be recovered from final samples
/// or coefficients alone. Preserve these regardless of the selected entropy
/// model; no old PJA flags or patch/edit scripts are represented here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScanReconstruction {
    pub padbit: i32,
    pub max_be: usize,
    pub ignore_ri: bool,
    pub wrap_pred: bool,
}
#[derive(Clone, Debug)]
pub(crate) struct Encoded {
    pub original_len: usize,
    pub kind: Kind,
    /// Raw marker/cut reconstruction bytes compressed by the shared raw model.
    /// Embedded APP JPEGs and MPO bytes after EOI remain here byte-exactly.
    pub header: Vec<u8>,
    pub scans: Vec<ScanReconstruction>,
    /// Near-lossless run side information for MED/Blend JPEG-LS only.
    pub runs: Option<Vec<u8>>,
    pub streams: Vec<Vec<u8>>,
}
impl Encoded {
    /// Internal comparison cost: data plus explicit field/length allowance.
    /// This is not a normative serialized size or a finalized Plus layout.
    pub fn accounted_size(&self) -> Result<usize> {
        let mut size = self
            .header
            .len()
            .checked_add(16)
            .ok_or(Error::Limit("codec payload size"))?;
        size = size
            .checked_add(
                self.scans
                    .len()
                    .checked_mul(16)
                    .ok_or(Error::Limit("scan reconstruction size"))?,
            )
            .ok_or(Error::Limit("codec payload size"))?;
        if let Some(runs) = &self.runs {
            size = size
                .checked_add(runs.len())
                .and_then(|n| n.checked_add(8))
                .ok_or(Error::Limit("run payload size"))?;
        }
        for s in &self.streams {
            size = size
                .checked_add(s.len())
                .and_then(|n| n.checked_add(8))
                .ok_or(Error::Limit("codec stream size"))?;
        }
        Ok(size)
    }
}
fn compressed_header(raw: &[u8], o: &Options) -> Result<Vec<u8>> {
    let mut cd = Encoder::new(o.limits.max_output_bytes);
    raw_model::code(&mut cd, &mut raw.to_vec(), o.limits.max_metadata_bytes)?;
    cd.finish()
}
fn samples(j: &jls::Image, blend: bool, o: &Options) -> Result<Vec<Vec<u8>>> {
    sample_model::encode_samples(j, blend, o.limits.max_output_bytes, &jls_limits(&o.limits))
}
fn decode_samples(
    j: &mut jls::Image,
    streams: &[Vec<u8>],
    blend: bool,
    l: &DecodeLimits,
) -> Result<()> {
    if streams.len() != sample_model::band_count(j) {
        return Err(Error::Invalid("sample band count"));
    }
    for (band, bytes) in streams.iter().enumerate() {
        let mut cd = Decoder::new(bytes);
        sample_model::code_band(&mut cd, j, band, blend, &jls_limits(l))?;
        cd.finish_check()?;
    }
    Ok(())
}
fn select(candidates: impl IntoIterator<Item = Result<Encoded>>, cap: usize) -> Result<Encoded> {
    let mut best: Option<Encoded> = None;
    let mut first_error = None;
    for candidate in candidates {
        match candidate {
            Ok(trial) => {
                let size = trial.accounted_size()?;
                if size > cap {
                    if first_error.is_none() {
                        first_error = Some(Error::Limit("codec encoded bytes"));
                    }
                    continue;
                }
                if best
                    .as_ref()
                    .is_none_or(|v| v.accounted_size().is_ok_and(|old| size < old))
                {
                    best = Some(trial);
                }
            }
            Err(error @ Error::Internal(_)) => return Err(error),
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    best.ok_or_else(|| first_error.unwrap_or(Error::Internal("no sample coding candidates")))
}
fn encode_jpeg(original: &[u8], o: &Options) -> Result<Encoded> {
    // Directly reject broken scans: no repair, edit script or whole-file raw
    // fallback. Valid alternate reconstruction decisions remain explicit.
    let j = jpeg::decompose_verified(original, &jpeg_limits(&o.limits))?;
    pixels(j.width, j.height, &o.limits)?;
    let header = compressed_header(&j.raw, o)?;
    let scans: Vec<ScanReconstruction> = j
        .scans
        .iter()
        .map(|s| ScanReconstruction {
            padbit: s.padbit,
            max_be: s.max_be,
            ignore_ri: s.ignore_ri,
            wrap_pred: s.wrap_pred,
        })
        .collect();
    if j.lossless {
        let view = lossless_view(&j);
        let modes = if o.blend {
            &[SampleMode::Med, SampleMode::Blend][..]
        } else {
            &[SampleMode::Med][..]
        };
        select(
            modes.iter().map(|&mode| {
                Ok(Encoded {
                    original_len: original.len(),
                    kind: Kind::JpegLossless(mode),
                    header: header.clone(),
                    scans: scans.clone(),
                    runs: None,
                    streams: samples(&view, mode == SampleMode::Blend, o)?,
                })
            }),
            o.limits.max_output_bytes,
        )
    } else {
        let mut cs = components(&j)?;
        let streams = match o.model {
            Model::V7 => model::encode(&cs, o.limits.max_output_bytes)?,
            Model::Full | Model::Fast => {
                let mut out = Vec::new();
                let mut remaining = o.limits.max_output_bytes;
                for band in 0..crate::legacy_model::band_count(&cs)? {
                    let mut cd = Encoder::new(remaining);
                    crate::legacy_model::code_band(&mut cd, &mut cs, band, o.model == Model::Fast)?;
                    let bytes = cd.finish()?;
                    remaining = remaining
                        .checked_sub(bytes.len())
                        .ok_or(Error::Limit("coefficient output"))?;
                    out.push(bytes);
                }
                out
            }
        };
        Ok(Encoded {
            original_len: original.len(),
            kind: Kind::JpegDct(o.model),
            header,
            scans,
            runs: None,
            streams,
        })
    }
}
fn encode_jls(original: &[u8], o: &Options) -> Result<Encoded> {
    let j = jls::verify(original, &jls_limits(&o.limits))?;
    pixels(j.width, j.height, &o.limits)?;
    let header = compressed_header(&j.raw, o)?;
    let modes = if o.blend {
        &[SampleMode::Med, SampleMode::Symbols, SampleMode::Blend][..]
    } else {
        &[SampleMode::Med, SampleMode::Symbols][..]
    };
    select(
        modes.iter().map(|&mode| {
            let (runs, streams) = if mode == SampleMode::Symbols {
                let mut cd = Encoder::new(o.limits.max_output_bytes);
                sample_model::code_symbol_stream(&mut cd, &mut j.clone(), &jls_limits(&o.limits))?;
                (None, vec![cd.finish()?])
            } else {
                let mut cd = Encoder::new(o.limits.max_output_bytes);
                sample_model::code_runs(&mut cd, &mut j.clone(), o.limits.max_hint_entries)?;
                (
                    Some(cd.finish()?),
                    samples(&j, mode == SampleMode::Blend, o)?,
                )
            };
            Ok(Encoded {
                original_len: original.len(),
                kind: Kind::JpegLs(mode),
                header: header.clone(),
                scans: Vec::new(),
                runs,
                streams,
            })
        }),
        o.limits.max_output_bytes,
    )
}
pub(crate) fn encode(original: &[u8], o: &Options) -> Result<Encoded> {
    if original.len() > o.limits.max_input_bytes {
        return Err(Error::Limit("codec input bytes"));
    }
    let encoded = if jls::is_jls(original) {
        encode_jls(original, o)?
    } else {
        encode_jpeg(original, o)?
    };
    if encoded.accounted_size()? > o.limits.max_output_bytes {
        return Err(Error::Limit("codec encoded bytes"));
    }
    let verify_limits = DecodeLimits {
        max_input_bytes: o.limits.max_output_bytes,
        ..o.limits.clone()
    };
    if o.verify && decode(&encoded, &verify_limits)? != original {
        return Err(Error::Internal("codec original-byte verification"));
    }
    Ok(encoded)
}
pub(crate) fn decode(encoded: &Encoded, l: &DecodeLimits) -> Result<Vec<u8>> {
    if encoded.original_len > l.max_output_bytes {
        return Err(Error::Limit("codec original bytes"));
    }
    if encoded.accounted_size()? > l.max_input_bytes {
        return Err(Error::Limit("codec encoded bytes"));
    }
    if encoded.header.len() < 4
        || encoded.streams.is_empty()
        || encoded.streams.len() > 16
        || encoded.streams.iter().any(|s| s.len() < 4)
    {
        return Err(Error::Invalid("codec stream dimensions"));
    }
    let mut raw = Vec::new();
    let mut cd = Decoder::new(&encoded.header);
    raw_model::code(
        &mut cd,
        &mut raw,
        l.max_metadata_bytes.min(l.max_output_bytes),
    )?;
    cd.finish_check()?;
    let out = match encoded.kind {
        Kind::JpegLs(mode) => {
            if !encoded.scans.is_empty() {
                return Err(Error::Invalid("JPEG-LS scan reconstruction fields"));
            }
            let mut j = jls::parse_headers(&raw, &jls_limits(l))?;
            pixels(j.width, j.height, l)?;
            if mode == SampleMode::Symbols {
                if encoded.runs.is_some() || encoded.streams.len() != 1 {
                    return Err(Error::Invalid("JPEG-LS symbol stream shape"));
                }
                let mut cd = Decoder::new(&encoded.streams[0]);
                sample_model::code_symbol_stream(&mut cd, &mut j, &jls_limits(l))?;
                cd.finish_check()?;
            } else {
                let runs = encoded
                    .runs
                    .as_ref()
                    .filter(|r| r.len() >= 4)
                    .ok_or(Error::Invalid("JPEG-LS run side information"))?;
                let mut cd = Decoder::new(runs);
                sample_model::code_runs(&mut cd, &mut j, l.max_hint_entries)?;
                cd.finish_check()?;
                decode_samples(&mut j, &encoded.streams, mode == SampleMode::Blend, l)?;
            }
            jls::recompose(&j, &jls_limits(l))?
        }
        Kind::JpegDct(mode) => {
            if encoded.runs.is_some() {
                return Err(Error::Invalid("DCT run side information"));
            }
            let mut j = jpeg::parse_headers(&raw, &jpeg_limits(l))?;
            pixels(j.width, j.height, l)?;
            if j.lossless {
                return Err(Error::Invalid("DCT frame has lossless samples"));
            }
            apply_scans(&mut j, &encoded.scans)?;
            let mut cs = components(&j)?;
            if encoded.streams.len() != model::band_count(&cs)? {
                return Err(Error::Invalid("DCT band count"));
            }
            let refs = encoded
                .streams
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>();
            match mode {
                Model::V7 => model::decode_strict(&mut cs, &refs)?,
                Model::Full | Model::Fast => {
                    for (band, bytes) in encoded.streams.iter().enumerate() {
                        let mut cd = Decoder::new(bytes);
                        crate::legacy_model::code_band(
                            &mut cd,
                            &mut cs,
                            band,
                            mode == Model::Fast,
                        )?;
                        cd.finish_check()?;
                    }
                }
            }
            install(&mut j, cs);
            jpeg::recompose(&j, &jpeg_limits(l))?
        }
        Kind::JpegLossless(mode) => {
            if mode == SampleMode::Symbols || encoded.runs.is_some() {
                return Err(Error::Invalid("lossless JPEG sample mode"));
            }
            let mut j = jpeg::parse_headers(&raw, &jpeg_limits(l))?;
            pixels(j.width, j.height, l)?;
            if !j.lossless {
                return Err(Error::Invalid("sample frame has DCT coefficients"));
            }
            apply_scans(&mut j, &encoded.scans)?;
            let mut view = lossless_view(&j);
            decode_samples(&mut view, &encoded.streams, mode == SampleMode::Blend, l)?;
            for (c, v) in j.components.iter_mut().zip(view.components) {
                c.samples = v.samples;
            }
            jpeg::recompose(&j, &jpeg_limits(l))?
        }
    };
    if out.len() != encoded.original_len {
        return Err(Error::Invalid("codec original length"));
    }
    Ok(out)
}
