// PJA MIT: pjacodec.pas J2kChoose/CompressJ2k/ReadJ2k algorithm, with owned
// internal records instead of exposing the original PJA archive syntax.
use super::{Codestream, codestream::Limits, modes, pixels};
use crate::{
    Error, Result,
    arith::{Decoder, Encoder},
    raw_model,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Decisions,
    Pixels,
    Ht,
    HtPixels,
}
#[derive(Clone, Debug)]
pub(crate) struct Part {
    pub location: Codestream,
    pub mode: Mode,
    pub data: Vec<u8>,
}
#[derive(Clone, Debug)]
pub(crate) struct Encoded {
    pub parts: Vec<Part>,
    pub host: Vec<u8>,
    pub original_size: usize,
}
fn decoded(data: &[u8], len: usize, mode: Mode, limits: &Limits) -> Result<Vec<u8>> {
    match mode {
        Mode::Decisions => modes::decode(data, len, false, limits),
        Mode::Ht => modes::decode(data, len, true, limits),
        Mode::Pixels => pixels::decode(data, len, false, limits),
        Mode::HtPixels => pixels::decode(data, len, true, limits),
    }
}
fn choose(data: &[u8], limits: &Limits) -> Result<(Mode, Vec<u8>)> {
    let ht = super::is_ht(data);
    let (decision, pixel) = if ht {
        (Mode::Ht, Mode::HtPixels)
    } else {
        (Mode::Decisions, Mode::Pixels)
    };
    let mut best: Option<(Mode, Vec<u8>)> = None;
    let mut error = None;
    let verify_limits = Limits {
        max_input_bytes: limits.max_output_bytes,
        ..limits.clone()
    };
    // Both coding tools remain candidates even when they expand the source.
    for (mode, result) in [
        (decision, modes::encode(data, ht, limits)),
        (pixel, pixels::encode(data, ht, limits)),
    ] {
        match result {
            Ok(bytes) => {
                let restored = decoded(&bytes, data.len(), mode, &verify_limits)?;
                if restored != data {
                    return Err(Error::Internal("J2K candidate byte verification"));
                }
                if best
                    .as_ref()
                    .is_none_or(|(_, previous)| bytes.len() < previous.len())
                {
                    best = Some((mode, bytes));
                }
            }
            Err(e @ Error::Internal(_)) => return Err(e),
            Err(e) => {
                fn rank(e: &Error) -> u8 {
                    match e {
                        Error::Limit(_) => 3,
                        Error::Invalid(_) | Error::Jpeg(_) => 2,
                        Error::Unsupported(_) => 1,
                        Error::Internal(_) => 4,
                    }
                }
                if error
                    .as_ref()
                    .is_none_or(|previous| rank(&e) > rank(previous))
                {
                    error = Some(e);
                }
            }
        }
    }
    best.ok_or_else(|| error.unwrap_or(Error::Unsupported("J2K coding tools")))
}
pub(crate) fn encode(data: &[u8], limits: &Limits) -> Result<Encoded> {
    if data.len() > limits.max_input_bytes {
        return Err(Error::Limit("J2K input bytes"));
    }
    let cs = super::codestreams(data).ok_or(Error::Unsupported("not a JPEG 2000 family file"))?;
    if cs.is_empty() {
        return Err(Error::Unsupported("JPEG 2000 file without codestreams"));
    }
    let mut parts = Vec::with_capacity(cs.len());
    let mut total = 0usize;
    let mut host = Vec::with_capacity(data.len());
    let mut pos = 0;
    for c in cs {
        let (mode, stream) = choose(&data[c.off..c.off + c.len], limits)?;
        total = total
            .checked_add(stream.len())
            .filter(|&n| n <= limits.max_output_bytes)
            .ok_or(Error::Limit("J2K modelled outputs"))?;
        host.extend_from_slice(&data[pos..c.off]);
        pos = c.off + c.len;
        parts.push(Part {
            location: c,
            mode,
            data: stream,
        });
    }
    host.extend_from_slice(&data[pos..]);
    let mut cd = Encoder::new(limits.max_output_bytes - total);
    raw_model::code(&mut cd, &mut host, limits.max_input_bytes)?;
    let host = cd.finish()?;
    let result = Encoded {
        parts,
        host,
        original_size: data.len(),
    };
    let verify_limits = Limits {
        max_input_bytes: limits.max_output_bytes,
        ..limits.clone()
    };
    if decode(&result, &verify_limits)? != data {
        return Err(Error::Internal("J2K whole-file verification"));
    }
    Ok(result)
}
pub(crate) fn decode(e: &Encoded, limits: &Limits) -> Result<Vec<u8>> {
    if e.parts.is_empty() {
        return Err(Error::Invalid("J2K missing structured codestreams"));
    }
    let input = e.parts.iter().try_fold(e.host.len(), |n, p| {
        n.checked_add(p.data.len())
            .ok_or(Error::Limit("J2K encoded size"))
    })?;
    if input > limits.max_input_bytes {
        return Err(Error::Limit("J2K encoded input bytes"));
    }
    if e.original_size > limits.max_output_bytes || e.parts.len() > 16384 {
        return Err(Error::Limit("J2K reconstructed file"));
    }
    let mut prev = 0;
    let mut cut = 0usize;
    for p in &e.parts {
        let c = p.location;
        let end = c
            .off
            .checked_add(c.len)
            .ok_or(Error::Limit("J2K location overflow"))?;
        if c.len == 0 || c.off < prev || end > e.original_size || p.data.is_empty() {
            return Err(Error::Invalid("J2K part layout"));
        }
        prev = end;
        cut = cut
            .checked_add(c.len)
            .ok_or(Error::Limit("J2K total cut"))?;
    }
    let mut host = Vec::new();
    let host_cap = e
        .original_size
        .checked_sub(cut)
        .ok_or(Error::Invalid("J2K host length"))?;
    let mut host_coder = Decoder::new(&e.host);
    raw_model::code(&mut host_coder, &mut host, host_cap)?;
    host_coder.finish_check()?;
    if host.len() != host_cap {
        return Err(Error::Invalid("J2K host size"));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(e.original_size)
        .map_err(|_| Error::Limit("J2K output allocation"))?;
    let mut hp = 0usize;
    for p in &e.parts {
        let c = p.location;
        let n = c
            .off
            .checked_sub(out.len())
            .ok_or(Error::Invalid("J2K insertion order"))?;
        let end = hp.checked_add(n).ok_or(Error::Limit("J2K host position"))?;
        out.extend_from_slice(host.get(hp..end).ok_or(Error::Invalid("J2K host prefix"))?);
        hp = end;
        out.extend_from_slice(&decoded(&p.data, c.len, p.mode, limits)?);
    }
    out.extend_from_slice(&host[hp..]);
    if out.len() != e.original_size {
        return Err(Error::Invalid("J2K reconstructed length"));
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::j2k::Kind;
    fn empty_codestream() -> Vec<u8> {
        let mut d = vec![255, 79, 255, 81, 0, 41, 0, 0];
        for value in [1u32, 1, 0, 0, 1, 1, 0, 0] {
            d.extend(value.to_be_bytes());
        }
        d.extend([0, 1, 7, 1, 1]);
        d.extend([255, 82, 0, 12, 0, 0, 0, 1, 0, 0, 4, 4, 0, 1]);
        d.extend([255, 92, 0, 4, 64, 64]);
        d.extend([
            255, 144, 0, 10, 0, 0, 0, 0, 0, 15, 0, 1, 255, 147, 0, 255, 217,
        ]);
        d
    }
    #[test]
    fn expanding_codestream_is_still_structurally_encoded() {
        let mut data = empty_codestream();
        // Deterministic high-entropy COM metadata makes the small image's
        // structural encoding unprofitable, while remaining a valid stream.
        let mut com = vec![255, 100, 0x10, 4, 0, 0];
        let mut random = 0x9365_1047u32;
        for _ in 0..4096 {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            com.push((random >> 24) as u8);
        }
        data.splice(45..45, com);
        let limits = Limits {
            max_input_bytes: data.len(),
            max_output_bytes: 65536,
            ..Limits::default()
        };
        let decisions = modes::encode(&data, false, &limits).unwrap();
        let pixels = pixels::encode(&data, false, &limits).unwrap();
        // This tiny source cannot amortize either model's structural overhead.
        assert!(decisions.len() > data.len());
        assert!(pixels.len() > data.len());
        let coded = encode(&data, &limits).unwrap();
        assert_eq!(coded.parts.len(), 1);
        assert_eq!(coded.parts[0].data.len(), decisions.len().min(pixels.len()));
        let decode_limits = Limits {
            max_input_bytes: limits.max_output_bytes,
            ..limits.clone()
        };
        assert_eq!(decode(&coded, &decode_limits).unwrap(), data);
    }
    #[test]
    fn failed_candidates_return_errors_without_raw_fallback() {
        // Family signature is recognized, but the SIZ marker is truncated.
        let malformed = [255, 79, 255, 81, 0, 41];
        assert!(matches!(
            encode(&malformed, &Limits::default()),
            Err(Error::Invalid(_))
        ));
        let data = empty_codestream();
        assert!(matches!(
            encode(
                &data,
                &Limits {
                    max_output_bytes: 0,
                    ..Limits::default()
                }
            ),
            Err(Error::Limit(_))
        ));
        let no_parts = Encoded {
            parts: Vec::new(),
            host: Vec::new(),
            original_size: 0,
        };
        assert!(matches!(
            decode(&no_parts, &Limits::default()),
            Err(Error::Invalid("J2K missing structured codestreams"))
        ));
    }
    #[test]
    #[ignore = "requires pinned upstream JPEG 2000 fixtures"]
    fn all_native_j2k_paths() {
        let root = std::path::PathBuf::from(std::env::var_os("RS_JXL_PLUS_PJA_FIXTURES").unwrap())
            .join("tests/j2k");
        let mut seen = [0; 4];
        let mut n = 0;
        let mut failures = Vec::new();
        for e in std::fs::read_dir(root).unwrap() {
            let path = e.unwrap().path();
            if !path.is_file() {
                continue;
            }
            let data = std::fs::read(&path).unwrap();
            if super::super::kind(&data) == Kind::None {
                continue;
            }
            let l = Limits {
                max_input_bytes: 1 << 26,
                max_output_bytes: 1 << 26,
                max_samples: 1 << 24,
                max_side_bytes: 1 << 26,
            };
            match encode(&data, &l) {
                Ok(coded) => {
                    assert_eq!(decode(&coded, &l).unwrap(), data);
                    for p in &coded.parts {
                        seen[p.mode as usize] += 1;
                    }
                    eprintln!(
                        "PASS {} {:?}",
                        path.display(),
                        coded.parts.iter().map(|p| p.mode).collect::<Vec<_>>()
                    );
                    n += 1;
                }
                Err(e) => failures.push(format!("{}: {e}", path.display())),
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
        assert!(n >= 19);
        assert!(
            seen.iter().all(|&n| n > 0),
            "all four modelling paths {seen:?}"
        );
    }
}
