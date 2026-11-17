use rs_jxl::{icc, jpeg};
use rs_jxl_plus::{DecodeLimits, EncodeOptions, Error, decode_jpeg, encode_jpeg};

#[allow(dead_code)]
#[path = "../../rs-jxl/tests/support/jpeg_fixtures.rs"]
mod fixtures;

#[test]
fn jpeg_modes_and_metadata_reconstruct_every_byte() {
    let options = EncodeOptions::default();
    for (name, original) in fixtures::fixtures()
        .into_iter()
        .chain(fixtures::metadata_fixtures())
    {
        let packed = encode_jpeg(&original, &options).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(packed.starts_with(rs_jxl_plus::MAGIC));
        assert_eq!(
            decode_jpeg(&packed, &options.limits).unwrap(),
            original,
            "{name}"
        );
    }
}

#[test]
fn bounded_container_and_hostile_lengths() {
    let original = fixtures::fixtures().remove(0).1;
    let options = EncodeOptions::default();
    let packed = encode_jpeg(&original, &options).unwrap();
    for end in 0..packed.len() {
        assert!(
            decode_jpeg(&packed[..end], &options.limits).is_err(),
            "prefix {end}"
        );
    }
    // Version 1 has a 40-byte fixed header and 140-byte components.
    // Exercise fields whose malformed value must fail structurally, rather
    // than relying on a hash to detect arbitrary content changes.
    for offset in [12, 16, 20, 28, 32, 40, 44, 48] {
        let mut corrupt = packed.clone();
        corrupt[offset..offset + 4].fill(255);
        assert!(
            decode_jpeg(&corrupt, &options.limits).is_err(),
            "offset {offset}"
        );
    }
    let mut version = packed.clone();
    version[8] = 255;
    assert!(matches!(
        decode_jpeg(&version, &options.limits),
        Err(Error::Unsupported(_))
    ));
    let mut old_version = packed.clone();
    old_version[8..10].copy_from_slice(&0u16.to_le_bytes());
    assert!(matches!(
        decode_jpeg(&old_version, &options.limits),
        Err(Error::Unsupported("format version"))
    ));
    let mut zero_quant = packed.clone();
    zero_quant[52..54].fill(0);
    assert!(matches!(
        decode_jpeg(&zero_quant, &options.limits),
        Err(Error::Invalid("zero quantization value"))
    ));
    let n = packed[28] as usize;
    let mut huge_stream = packed.clone();
    let length_offset = 40 + 140 * n;
    huge_stream[length_offset..length_offset + 8].fill(255);
    assert!(decode_jpeg(&huge_stream, &options.limits).is_err());
    let mut appended = packed.clone();
    appended.push(0);
    assert!(decode_jpeg(&appended, &options.limits).is_err());
    for limits in [
        DecodeLimits {
            max_input_bytes: packed.len() - 1,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_output_bytes: original.len() - 1,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_coefficient_bytes: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_metadata_bytes: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_pixels: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_block_visits: 0,
            ..DecodeLimits::default()
        },
    ] {
        assert!(matches!(
            decode_jpeg(&packed, &limits),
            Err(Error::Limit(_))
        ));
    }
    assert!(matches!(
        encode_jpeg(
            &original,
            &EncodeOptions {
                limits: DecodeLimits {
                    max_pixels: 0,
                    ..DecodeLimits::default()
                },
                ..EncodeOptions::default()
            }
        ),
        Err(Error::Limit("pixels"))
    ));
}

#[test]
fn declared_original_length_and_hashless_layout() {
    let original = fixtures::fixtures().remove(0).1;
    let options = EncodeOptions::default();
    let packed = encode_jpeg(&original, &options).unwrap();
    assert_eq!(u16::from_le_bytes(packed[8..10].try_into().unwrap()), 1);
    assert_eq!(
        u64::from_le_bytes(packed[20..28].try_into().unwrap()),
        original.len() as u64
    );
    let components = packed[28] as usize;
    let bands = packed[29] as usize;
    let metadata = u64::from_le_bytes(packed[32..40].try_into().unwrap()) as usize;
    let lengths = 40 + 140 * components;
    let stream_bytes = (0..bands)
        .map(|i| {
            u64::from_le_bytes(
                packed[lengths + i * 8..lengths + (i + 1) * 8]
                    .try_into()
                    .unwrap(),
            ) as usize
        })
        .sum::<usize>();
    assert_eq!(
        packed.len(),
        40 + 140 * components + 8 * bands + metadata + stream_bytes,
        "no original hash or trailing digest bytes"
    );
    let mut wrong_length = packed.clone();
    wrong_length[20..28].copy_from_slice(&(original.len() as u64 + 1).to_le_bytes());
    assert!(matches!(
        decode_jpeg(&wrong_length, &options.limits),
        Err(Error::Invalid("original JPEG size"))
    ));
    // Version 1 intentionally makes no promise that an arbitrary payload
    // modification is rejected or reconstructs the former input bytes.
}

#[test]
#[ignore = "uses explicit local seven-case manifest via RS_JXL_PLUS_JPEG_MANIFEST"]
fn local_manifest_original_bytes() {
    let manifest = std::env::var_os("RS_JXL_PLUS_JPEG_MANIFEST").expect("set manifest path");
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let cases = value
        .as_array()
        .or_else(|| value.get("cases").and_then(|v| v.as_array()))
        .expect("manifest cases");
    assert!(
        cases.len() <= 7,
        "large corpus requires an explicit separate run"
    );
    for case in cases {
        let path = case
            .as_str()
            .or_else(|| case.get("original").and_then(|v| v.as_str()))
            .or_else(|| case.get("path").and_then(|v| v.as_str()))
            .or_else(|| case.get("input").and_then(|v| v.as_str()))
            .expect("case input path");
        let original = std::fs::read(path).unwrap();
        let start = std::time::Instant::now();
        let packed = encode_jpeg(&original, &EncodeOptions::default())
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        let encode_seconds = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        assert_eq!(
            decode_jpeg(&packed, &DecodeLimits::default()).unwrap(),
            original,
            "{path}"
        );
        eprintln!(
            "{} original={} jxl_plus={} verified_encode_s={encode_seconds:.3} decode_s={:.3}",
            std::path::Path::new(path)
                .file_name()
                .unwrap()
                .to_string_lossy(),
            original.len(),
            packed.len(),
            start.elapsed().as_secs_f64()
        );
    }
}
