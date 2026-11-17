//! Internal composition of the translated PJA algorithms. These owned records
//! are deliberately not a public or stable on-disk format. The final user
//! requested JXL codestream plus extensions is a subsequent integration step.
use crate::{DecodeLimits, Error, Result, codec, j2k};
#[derive(Clone, Debug)]
pub(crate) enum Encoded {
    Jpeg(codec::Encoded),
    Jpeg2000(j2k::pipeline::Encoded),
}
fn j2k_limits(l: &DecodeLimits) -> j2k::codestream::Limits {
    j2k::codestream::Limits {
        max_input_bytes: l.max_input_bytes,
        max_output_bytes: l.max_output_bytes,
        max_samples: (l.max_coefficient_bytes / 2) as u64,
        max_side_bytes: l.max_metadata_bytes,
    }
}
pub(crate) fn encode(data: &[u8], limits: &DecodeLimits) -> Result<Encoded> {
    if data.len() > limits.max_input_bytes {
        return Err(Error::Limit("ported input bytes"));
    }
    if j2k::kind(data) != j2k::Kind::None {
        return Ok(Encoded::Jpeg2000(j2k::pipeline::encode(
            data,
            &j2k_limits(limits),
        )?));
    }
    let options = codec::Options {
        limits: limits.clone(),
        ..Default::default()
    };
    Ok(Encoded::Jpeg(codec::encode(data, &options)?))
}
pub(crate) fn decode(data: &Encoded, limits: &DecodeLimits) -> Result<Vec<u8>> {
    match data {
        Encoded::Jpeg2000(e) => j2k::pipeline::decode(e, &j2k_limits(limits)),
        Encoded::Jpeg(e) => codec::decode(e, limits),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned upstream codec regression fixtures"]
    fn all_ported_codecs() {
        let root = std::path::PathBuf::from(std::env::var_os("RS_JXL_PLUS_PJA_FIXTURES").unwrap())
            .join("tests");
        let limits = DecodeLimits::default();
        let mut n = 0;
        let mut rejected = 0;
        let mut failures = Vec::new();
        for folder in ["jpg", "jls", "j2k"] {
            for f in std::fs::read_dir(root.join(folder)).unwrap() {
                let path = f.unwrap().path();
                if !path.is_file() {
                    continue;
                }
                let bytes = std::fs::read(&path).unwrap();
                if !bytes.starts_with(b"\xff\xd8") && j2k::kind(&bytes) == j2k::Kind::None {
                    continue;
                }
                // These upstream cases previously needed the removed generic
                // repair/patch or whole-file raw fallback. Their rejection is
                // now explicit, while valid source reconstruction stays exact.
                if folder == "jpg"
                    && [
                        "byte_before_eoi.jpg",
                        "cut_short.jpg",
                        "prog_restarts_cut.jpg",
                        "restarts_misnumbered.jpg",
                        "scan_ends_early.jpg",
                    ]
                    .iter()
                    .any(|&name| path.file_name().is_some_and(|n| n == name))
                {
                    assert!(encode(&bytes, &limits).is_err(), "{}", path.display());
                    rejected += 1;
                    continue;
                }
                let result = encode(&bytes, &limits).and_then(|e| decode(&e, &limits));
                match result {
                    Ok(v) => {
                        assert_eq!(v, bytes, "{}", path.display());
                        n += 1;
                    }
                    Err(e) => failures.push(format!("{}: {e}", path.display())),
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
        assert_eq!(n, 61);
        assert_eq!(rejected, 5);
        eprintln!(
            "{n} valid originals reconstructed exactly; {rejected} repair-only inputs rejected"
        );
    }
}
