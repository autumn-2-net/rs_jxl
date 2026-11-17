use super::*;
fn marker(d: &mut Vec<u8>, m: u8, payload: &[u8]) {
    d.extend([255, m]);
    d.extend(((payload.len() + 2) as u16).to_be_bytes());
    d.extend(payload);
}
fn headers(
    sof: u8,
    precision: u8,
    width: u16,
    height: u16,
    scans: &[(u8, u8, u8)],
    ri: u16,
) -> Vec<u8> {
    let mut d = vec![255, 216];
    let mut q = vec![0];
    q.extend([1; 64]);
    marker(&mut d, 219, &q);
    let mut dc = vec![0];
    dc.extend([0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    dc.extend(0..17);
    let mut ac = vec![16];
    ac.extend([0, 0, 0, 0, 0, 0, 0, 255, 1, 0, 0, 0, 0, 0, 0, 0]);
    ac.extend(0..=255);
    dc.extend(ac);
    marker(&mut d, 196, &dc);
    let mut frame = vec![precision];
    frame.extend(height.to_be_bytes());
    frame.extend(width.to_be_bytes());
    frame.extend([1, 1, 17, 0]);
    marker(&mut d, sof, &frame);
    if ri > 0 {
        marker(&mut d, 221, &ri.to_be_bytes());
    }
    for &(ss, se, ahal) in scans {
        marker(&mut d, 218, &[1, 1, 0, ss, se, ahal]);
    }
    d.extend([255, 217]);
    d
}
#[test]
fn known_zero_huffman_bytes() {
    let mut d = vec![255, 216];
    let mut dc = vec![0, 1];
    dc.extend([0; 15]);
    dc.push(0);
    let mut ac = vec![16, 1];
    ac.extend([0; 15]);
    ac.push(0);
    dc.extend(ac);
    marker(&mut d, 196, &dc);
    marker(&mut d, 192, &[8, 0, 8, 0, 8, 1, 1, 17, 0]);
    marker(&mut d, 218, &[1, 1, 0, 0, 63, 0]);
    d.extend([0x3f, 255, 217]);
    let j = decompose_verified(&d, &Limits::default()).unwrap();
    assert!(j.components[0].coef.iter().all(|&x| x == 0));
    assert_eq!(j.scans[0].padbit, 1);
}
#[test]
fn known_lossless_midpoint_bytes() {
    let mut d = vec![255, 216];
    let mut dc = vec![0, 1];
    dc.extend([0; 15]);
    dc.push(0);
    marker(&mut d, 196, &dc);
    marker(&mut d, 195, &[8, 0, 1, 0, 1, 1, 1, 17, 0]);
    marker(&mut d, 218, &[1, 1, 0, 1, 0, 0]);
    d.extend([0x7f, 255, 217]);
    let j = decompose_verified(&d, &Limits::default()).unwrap();
    assert_eq!(j.components[0].samples, [128]);
}
#[test]
fn all_dct_coding_modes_and_precision() {
    for sof in [192, 193, 194, 201, 202] {
        for precision in [8, 12] {
            let progressive = sof == 194 || sof == 202;
            let scans = if progressive {
                vec![(0, 0, 1), (1, 63, 1), (0, 0, 16), (1, 63, 16)]
            } else {
                vec![(0, 63, 0)]
            };
            let raw = headers(sof, precision, 24, 24, &scans, 2);
            let mut j = parse_headers(&raw, &Limits::default()).unwrap();
            for (b, cf) in j.components[0].coef.chunks_exact_mut(64).enumerate() {
                cf[0] = (b as i16 * 17) - 65;
                for (k, c) in cf.iter_mut().enumerate().skip(1) {
                    if (k + b) % 7 == 0 {
                        *c = if k % 2 == 0 {
                            (k + b) as i16
                        } else {
                            -((k + b) as i16)
                        };
                    }
                }
            }
            let original = j.components[0].coef.clone();
            let bytes = recompose(&j, &Limits::default()).unwrap();
            let parsed = decompose_verified(&bytes, &Limits::default())
                .unwrap_or_else(|e| panic!("SOF {sof:x} precision {precision}: {e}"));
            assert_eq!(
                parsed.components[0].coef, original,
                "SOF {sof:x} precision {precision}"
            );
        }
    }
}
#[test]
fn lossless_all_predictors_point_transform_and_restarts() {
    for precision in [2, 8, 12, 16] {
        for psv in 1..=7 {
            for al in [0, 1] {
                let raw = headers(195, precision, 7, 5, &[(psv, 0, al)], 14);
                let mut j = parse_headers(&raw, &Limits::default()).unwrap();
                let top = (1u32 << (precision - al)) - 1;
                for (k, v) in j.components[0].samples.iter_mut().enumerate() {
                    *v = ((k as u32 * 8191 + 17) & top) as u16;
                }
                let expected = j.components[0].samples.clone();
                let bytes = recompose(&j, &Limits::default()).unwrap();
                let parsed = decompose_verified(&bytes, &Limits::default())
                    .unwrap_or_else(|e| panic!("P{precision} predictor{psv} Pt{al}: {e}"));
                assert_eq!(parsed.components[0].samples, expected);
            }
        }
    }
}
#[test]
fn dnl_ignored_restart_and_metadata_tail() {
    let raw = headers(192, 8, 16, 8, &[(0, 63, 0)], 1);
    let mut j = parse_headers(&raw, &Limits::default()).unwrap();
    j.scans[0].ri = 0;
    let mut bytes = recompose(&j, &Limits::default()).unwrap();
    bytes.extend(b"post EOI retained");
    let parsed = decompose_verified(&bytes, &Limits::default()).unwrap();
    assert!(parsed.scans[0].ignore_ri);
    let mut j = parsed;
    let sof = j.raw.windows(2).position(|w| w == [255, 192]).unwrap();
    j.raw[sof + 5] = 0;
    j.raw[sof + 6] = 0;
    let cut = j.scans[0].cut;
    let dnl = [255, 220, 0, 4, 0, 8];
    j.raw.splice(cut..cut, dnl);
    let bytes = recompose(&j, &Limits::default()).unwrap();
    let parsed = decompose_verified(&bytes, &Limits::default()).unwrap();
    assert!(parsed.dnl);
    assert!(parsed.raw.ends_with(b"post EOI retained"));
}
#[test]
fn parser_resource_limits() {
    let raw = headers(192, 8, 24, 24, &[(0, 63, 0)], 0);
    let limits = Limits {
        max_blocks: 8,
        ..Limits::default()
    };
    assert!(matches!(parse_headers(&raw, &limits), Err(Error::Limit(_))));
    let j = parse_headers(&raw, &Limits::default()).unwrap();
    assert!(matches!(
        recompose(
            &j,
            &Limits {
                max_output_bytes: 10,
                ..Limits::default()
            }
        ),
        Err(Error::Limit(_))
    ));
}

#[test]
#[ignore = "independent upstream JPEG fixture files supplied by RS_JXL_PLUS_PJA_FIXTURES"]
fn upstream_fixture_roundtrips() {
    fn collect(path: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if path.is_dir() {
            for entry in std::fs::read_dir(path).unwrap() {
                collect(&entry.unwrap().path(), out);
            }
        } else if path.extension().is_some_and(|e| {
            matches!(
                e.to_string_lossy().to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "jpe"
            )
        }) {
            out.push(path.to_owned());
        }
    }
    let root = std::path::PathBuf::from(
        std::env::var_os("RS_JXL_PLUS_PJA_FIXTURES")
            .expect("set RS_JXL_PLUS_PJA_FIXTURES to independently generated JPEG fixtures"),
    );
    let mut paths = Vec::new();
    collect(&root, &mut paths);
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no JPEG fixtures under {}",
        root.display()
    );
    let mut failed = Vec::new();
    let mut passed = 0;
    for path in paths {
        let bytes = std::fs::read(&path).unwrap();
        match decompose_verified(&bytes, &Limits::default()) {
            Ok(j) => {
                passed += 1;
                eprintln!(
                    "PASS {}: {}x{} P{} lossless={} arithmetic={} progressive={} scans={} maxBE={:?} ignoreRI={} wrapPred={}",
                    path.display(),
                    j.width,
                    j.height,
                    j.precision,
                    j.lossless,
                    j.arithmetic,
                    j.progressive,
                    j.scans.len(),
                    j.scans.iter().map(|s| s.max_be).collect::<Vec<_>>(),
                    has_ignored_restarts(&j),
                    has_wrapped_predictor(&j)
                );
            }
            Err(e) => {
                // Deliberately damaged upstream inputs are rejected. Assert
                // their precise parser failure so this never hides a new
                // valid-input regression behind the removed repair fallback.
                let expected = match path.file_name().and_then(|s| s.to_str()) {
                    Some("byte_before_eoi.jpg") => Some("DataAfterScan@1746"),
                    Some("cut_short.jpg") => Some("Truncated@1797"),
                    Some("prog_restarts_cut.jpg") => Some("Truncated@2030"),
                    Some("restarts_misnumbered.jpg") => Some("BadRst@898"),
                    Some("scan_ends_early.jpg") => Some("Truncated@15637"),
                    _ => None,
                };
                if expected
                    .is_some_and(|expected| matches!(&e,Error::Jpeg(actual)if actual==expected))
                {
                    eprintln!("REJECTED damaged input {}: {e}", path.display());
                } else {
                    failed.push(format!("{}: {e}", path.display()));
                }
            }
        }
    }
    for error in &failed {
        eprintln!("FAIL {error}");
    }
    assert!(
        failed.is_empty(),
        "{passed} fixture roundtrips passed, {} failed",
        failed.len()
    );
}
