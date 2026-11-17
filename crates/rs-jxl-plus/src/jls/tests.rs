//! PJA JPEG-LS focused fixtures; see LICENSE-PJA/NOTICE-PJA.
use super::*;
#[path = "pinned_fixture.rs"]
mod pinned;
fn marker(out: &mut Vec<u8>, m: u8, payload: &[u8]) {
    out.extend_from_slice(&[255, m]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}
pub(crate) fn fixture(bits: usize, n: usize, ilv: i32, near: i32, ri: i32) -> Image {
    let (width, height) = (13usize, 5usize);
    let maxval = (1i32 << bits) - 1;
    let (t1, t2, t3) = default_thresholds(maxval, near);
    let components: Vec<_> = (0..n)
        .map(|c| Component {
            id: (c + 1) as u8,
            h: 1,
            v: 1,
            width,
            height,
            samples: (0..width * height)
                .map(|i| {
                    if near > 0 {
                        0
                    } else {
                        ((i * 57 + c * 113 + i / width * 37) as u32 % (maxval as u32 + 1)) as u16
                    }
                })
                .collect(),
        })
        .collect();
    let mut raw = vec![255, 216];
    let mut sof = vec![bits as u8, 0, height as u8, 0, width as u8, n as u8];
    for c in 0..n {
        sof.extend_from_slice(&[(c + 1) as u8, 0x11, 0]);
    }
    marker(&mut raw, 0xf7, &sof);
    if ri > 0 {
        marker(&mut raw, 0xdd, &(ri as u16).to_be_bytes());
    }
    let groups: Vec<Vec<usize>> = if ilv == 0 {
        (0..n).map(|c| vec![c]).collect()
    } else {
        vec![(0..n).collect()]
    };
    let mut scans = Vec::new();
    for indices in groups {
        let mut sos = vec![indices.len() as u8];
        for &c in &indices {
            sos.extend_from_slice(&[(c + 1) as u8, 0]);
        }
        sos.extend_from_slice(&[near as u8, ilv as u8, 0]);
        marker(&mut raw, 0xda, &sos);
        let runs = if near > 0 {
            vec![width as i32; height * if ilv == 2 { 1 } else { indices.len() }]
        } else {
            Vec::new()
        };
        scans.push(Scan {
            components: indices,
            near,
            ilv,
            maxval,
            t1,
            t2,
            t3,
            reset: 64,
            ri,
            cut: raw.len(),
            runs,
        });
    }
    raw.extend_from_slice(&[255, 217]);
    Image {
        width,
        height,
        bits,
        components,
        scans,
        raw,
    }
}
#[test]
fn modes_precisions_restarts_and_near_runs() {
    let limits = Limits::default();
    for (bits, n, ilv, near, ri) in [
        (2, 1, 0, 0, 0),
        (8, 1, 0, 0, 2),
        (16, 1, 0, 0, 0),
        (12, 3, 0, 0, 0),
        (8, 3, 1, 0, 2),
        (16, 4, 2, 0, 2),
        (8, 3, 2, 1, 2),
        (8, 3, 1, 2, 0),
    ] {
        let j = fixture(bits, n, ilv, near, ri);
        let bytes = recompose(&j, &limits).unwrap();
        let decoded = verify(&bytes, &limits).unwrap();
        assert_eq!(
            decoded.components, j.components,
            "{bits}/{n}/{ilv}/{near}/{ri}"
        );
        let headers = parse_headers(&decoded.raw, &limits).unwrap();
        assert_eq!(headers.scans.len(), j.scans.len());
        assert!(
            headers
                .components
                .iter()
                .all(|c| c.samples.iter().all(|&x| x == 0))
        );
    }
}
#[test]
fn pinned_original_near_sample_fixture() {
    let limits = Limits::default();
    let j = verify(pinned::PINNED_JLS, &limits).unwrap();
    assert_eq!(
        (j.width, j.height, j.bits, j.components.len()),
        (17, 13, 8, 3)
    );
    assert_eq!((j.scans[0].ilv, j.scans[0].near), (2, 1));
    assert!(!j.scans[0].runs.is_empty());
    for blend in [false, true] {
        let streams = crate::sample_model::encode_samples(&j, blend, 1 << 20, &limits).unwrap();
        let mut decoded = parse_headers(&j.raw, &limits).unwrap();
        decoded.scans[0].runs = j.scans[0].runs.clone();
        crate::sample_model::decode_samples(
            &mut decoded,
            &streams.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            blend,
            &limits,
        )
        .unwrap();
        assert_eq!(decoded.components, j.components);
        assert_eq!(recompose(&decoded, &limits).unwrap(), pinned::PINNED_JLS);
    }
}
#[test]
fn metadata_mapping_presets_and_trailing_bytes() {
    let limits = Limits::default();
    let mut j = fixture(8, 1, 0, 0, 0);
    let old = j.raw.len();
    let mut extra = Vec::new();
    marker(&mut extra, 0xfe, b"opaque comment");
    marker(&mut extra, 0xf8, &[2, 1, 1, 9]);
    marker(&mut extra, 0xf8, &[3, 1, 8]);
    j.raw.splice(2..2, extra);
    let delta = j.raw.len() - old;
    for s in &mut j.scans {
        s.cut += delta;
    }
    j.raw.extend_from_slice(b"tail");
    let bytes = recompose(&j, &limits).unwrap();
    assert_eq!(verify(&bytes, &limits).unwrap().raw, j.raw);
}
#[test]
fn malformed_truncated_and_budget() {
    let limits = Limits::default();
    let bytes = recompose(&fixture(8, 1, 0, 0, 0), &limits).unwrap();
    for stop in 0..bytes.len() {
        assert!(
            decompose(&bytes[..stop], &limits).is_err(),
            "truncation {stop}"
        );
    }
    assert!(
        decompose(
            &bytes,
            &Limits {
                max_samples: 1,
                ..limits.clone()
            }
        )
        .is_err()
    );
    let mut bad = fixture(8, 1, 0, 0, 0);
    bad.scans[0].cut = usize::MAX;
    assert!(recompose(&bad, &limits).is_err());
}

#[test]
#[ignore = "requires pinned upstream tests/jls in RS_JXL_PLUS_PJA_FIXTURES"]
fn upstream_jls_byte_reconstruction() {
    let root = std::path::PathBuf::from(
        std::env::var_os("RS_JXL_PLUS_PJA_FIXTURES")
            .expect("set RS_JXL_PLUS_PJA_FIXTURES to upstream root"),
    )
    .join("tests/jls");
    let mut pending = vec![root];
    let mut files = Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            let kind = entry.file_type().unwrap();
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file()
                && entry
                    .path()
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("jls"))
            {
                files.push(entry.path());
            }
        }
    }
    files.sort();
    assert!(!files.is_empty(), "no upstream JPEG-LS fixtures found");
    let limits = Limits::default();
    let mut supported = 0;
    for path in files {
        let data = std::fs::read(&path).unwrap();
        let j = verify(&data, &limits).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        supported += 1;
        assert_eq!(recompose(&j, &limits).unwrap(), data, "{}", path.display());
        for blend in [false, true] {
            let streams = crate::sample_model::encode_samples(&j, blend, 1 << 24, &limits).unwrap();
            let mut back = parse_headers(&j.raw, &limits).unwrap();
            for (s, original) in back.scans.iter_mut().zip(&j.scans) {
                s.runs = original.runs.clone();
            }
            crate::sample_model::decode_samples(
                &mut back,
                &streams.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                blend,
                &limits,
            )
            .unwrap();
            assert_eq!(
                back.components,
                j.components,
                "{} blend={blend}",
                path.display()
            );
        }
        let mut cd = crate::arith::Encoder::new(1 << 24);
        let mut encoded = j.clone();
        crate::sample_model::code_symbol_stream(&mut cd, &mut encoded, &limits).unwrap();
        let bytes = cd.finish().unwrap();
        let mut decoded = parse_headers(&j.raw, &limits).unwrap();
        let mut cd = crate::arith::Decoder::new(&bytes);
        crate::sample_model::code_symbol_stream(&mut cd, &mut decoded, &limits).unwrap();
        assert_eq!(
            decoded.components,
            j.components,
            "{} symbols",
            path.display()
        );
        assert_eq!(
            decoded.scans.iter().map(|s| &s.runs).collect::<Vec<_>>(),
            j.scans.iter().map(|s| &s.runs).collect::<Vec<_>>()
        );
    }
    assert!(supported > 0, "no valid upstream JPEG-LS fixtures passed");
}
