//! Typed coding-tool regressions. No PJA archive/interoperability protocol.
use super::*;
fn fixture() -> jls::Image {
    jls::tests::fixture(8, 3, 1, 1, 0)
}
#[test]
fn normal_jls_and_typed_structure_limits() {
    let l = DecodeLimits::default();
    let original = jls::recompose(&fixture(), &jls_limits(&l)).unwrap();
    let encoded = encode(&original, &Options::default()).unwrap();
    assert!(matches!(encoded.kind, Kind::JpegLs(_)));
    assert!(encoded.scans.is_empty());
    assert_eq!(decode(&encoded, &l).unwrap(), original);
    let mut wrong = encoded.clone();
    wrong.original_len += 1;
    assert!(matches!(
        decode(&wrong, &l),
        Err(Error::Invalid("codec original length"))
    ));
    let mut truncated = encoded.clone();
    truncated.header.pop();
    assert!(decode(&truncated, &l).is_err());
    let mut trailing = encoded.clone();
    trailing.streams[0].push(0);
    assert!(decode(&trailing, &l).is_err());
    let mut missing = encoded.clone();
    missing.streams.clear();
    assert!(decode(&missing, &l).is_err());
    let cap = encoded.accounted_size().unwrap();
    assert!(matches!(
        decode(
            &encoded,
            &DecodeLimits {
                max_input_bytes: cap - 1,
                ..l.clone()
            }
        ),
        Err(Error::Limit("codec encoded bytes"))
    ));
    assert!(matches!(
        encode(
            &original,
            &Options {
                limits: DecodeLimits {
                    max_output_bytes: 1,
                    ..l.clone()
                },
                ..Options::default()
            }
        ),
        Err(Error::Limit(_))
    ));
}
#[test]
fn selection_keeps_other_tools_but_exposes_internal_errors() {
    let e = Encoded {
        original_len: 1,
        kind: Kind::JpegLs(SampleMode::Symbols),
        header: vec![0; 4],
        scans: Vec::new(),
        runs: None,
        streams: vec![vec![0; 4]],
    };
    assert!(select([Err(Error::Unsupported("candidate")), Ok(e.clone())], 100).is_ok());
    assert!(matches!(
        select([Ok(e), Err(Error::Internal("broken model"))], 100),
        Err(Error::Internal("broken model"))
    ));
}
#[test]
fn one_input_budget_is_distinct_from_encoded_verification_budget() {
    let original = jls::recompose(&fixture(), &jls_limits(&DecodeLimits::default())).unwrap();
    let o = Options {
        limits: DecodeLimits {
            max_input_bytes: original.len(),
            ..DecodeLimits::default()
        },
        ..Options::default()
    };
    let encoded = encode(&original, &o).unwrap();
    let l = DecodeLimits {
        max_input_bytes: o.limits.max_output_bytes,
        ..o.limits.clone()
    };
    assert_eq!(decode(&encoded, &l).unwrap(), original);
}

fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var_os("RS_JXL_PLUS_PJA_FIXTURES").expect("set pinned upstream fixture root"),
    )
    .join("tests")
}
#[test]
#[ignore = "requires bounded pinned JPEG/JPEG-LS regression fixtures"]
fn legal_jpeg_jls_and_broken_scan_rejection() {
    let root = root();
    let options = Options::default();
    let broken = [
        "byte_before_eoi.jpg",
        "cut_short.jpg",
        "prog_restarts_cut.jpg",
        "restarts_misnumbered.jpg",
        "scan_ends_early.jpg",
    ];
    let mut legal = 0;
    let mut rejected = 0;
    let mut errors = Vec::new();
    for dir in ["jpg", "jls"] {
        let mut paths = std::fs::read_dir(root.join(dir))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .is_some_and(|e| e == "jpg" || e == "mpo" || e == "jls")
            })
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            let name = path.file_name().unwrap().to_str().unwrap();
            let original = std::fs::read(&path).unwrap();
            if broken.contains(&name) {
                let direct = jpeg::decompose_verified(&original, &jpeg_limits(&options.limits));
                let result = encode(&original, &options);
                match (direct, result) {
                    (Err(a), Err(b)) => {
                        assert_eq!(
                            a.to_string(),
                            b.to_string(),
                            "{name}: direct parser error must propagate, without repair or raw fallback"
                        );
                        rejected += 1;
                    }
                    _ => errors.push(format!("{name}: damaged scan accepted")),
                }
                continue;
            }
            match encode(&original, &options).and_then(|e| decode(&e, &options.limits)) {
                Ok(bytes) => {
                    assert_eq!(bytes, original, "{name}");
                    legal += 1;
                }
                Err(e) => errors.push(format!("{name}: {e}")),
            }
        }
    }
    assert!(
        errors.is_empty(),
        "{} legal, {} rejected: {errors:?}",
        legal,
        rejected
    );
    assert_eq!(rejected, 5);
    assert_eq!(legal, 42);
    eprintln!(
        "typed pipeline: {legal} legal JPEG/JPEG-LS originals reconstructed; {rejected} unrepaired damaged scans rejected"
    );
}
#[test]
#[ignore = "requires pinned native coefficient/sample tool fixtures"]
fn every_model_and_original_reconstruction_fields() {
    let root = root();
    let limits = DecodeLimits::default();
    for name in [
        "baseline_color.jpg",
        "arith_prog.jpg",
        "twelve_bit.jpg",
        "dri_no_restarts.jpg",
        "prog_correction_limit.jpg",
        "lossless_wrap.jpg",
        "stereo.mpo",
        "thumbs_twice.jpg",
        "two_images.jpg",
        "gain_map.jpg",
    ] {
        let original = std::fs::read(root.join("jpg").join(name)).unwrap();
        for model in [Model::Full, Model::Fast, Model::V7] {
            let encoded = encode(
                &original,
                &Options {
                    model,
                    ..Options::default()
                },
            )
            .unwrap_or_else(|e| panic!("{name} {model:?}: {e}"));
            assert_eq!(
                decode(&encoded, &limits).unwrap(),
                original,
                "{name} {model:?}"
            );
            if name == "dri_no_restarts.jpg" {
                assert!(encoded.scans.iter().any(|s| s.ignore_ri));
            }
            if name == "prog_correction_limit.jpg" {
                assert!(
                    encoded
                        .scans
                        .iter()
                        .any(|s| s.max_be != jpeg::LIBJPEG_MAX_BE)
                );
            }
            if name == "lossless_wrap.jpg" {
                assert!(encoded.scans.iter().any(|s| s.wrap_pred));
            }
        }
    }
}
