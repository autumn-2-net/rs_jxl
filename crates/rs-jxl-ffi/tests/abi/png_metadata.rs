use super::{call, encode, json, ok, spec};

fn box_bytes(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut bytes = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
    bytes.extend_from_slice(kind);
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn png_export_rejects_metadata_before_and_after_the_codestream() {
    let encoded = encode(spec("u8", 1, 2), &[7, 251]);
    assert_eq!(&encoded[4..8], b"JXL ");
    for kind in [b"xml ", b"Exif", b"jumb", b"jhgm", b"jbrd"] {
        for payload in [b"\0\0\0\0metadata".as_slice(), b""] {
            for position in [32, encoded.len()] {
                let mut input = encoded.clone();
                input.splice(position..position, box_bytes(kind, payload));
                let (info, _) = ok(json!({"operation":"info"}), &input);
                assert_eq!(info["boxes"][0]["type"], std::str::from_utf8(kind).unwrap());
                assert_eq!(info["boxes"][0]["data"], json!(payload));
                let (code, meta, output) = call(json!({"operation":"decode_to_png"}), &input);
                assert_eq!(code, 1, "{kind:?} at {position}: {meta}");
                assert!(
                    meta["error"]
                        .as_str()
                        .unwrap()
                        .contains("ancillary JXL boxes")
                );
                assert!(output.is_empty());
            }
        }
    }
}

#[test]
fn png_export_checks_complete_metadata_budgets_and_truncation() {
    let encoded = encode(spec("u8", 1, 2), &[7, 251]);
    let mut input = encoded.clone();
    input.extend(box_bytes(b"xml ", b"<xmp/>"));
    input.extend(box_bytes(b"xml ", b"<xmp/>"));
    let (code, meta, output) = call(
        json!({"operation":"decode_to_png","options":{"limits":{"max_metadata_bytes":11}}}),
        &input,
    );
    assert_eq!(code, 1, "{meta}");
    assert_eq!(meta["error_kind"], "resource_limit");
    assert!(output.is_empty());
    for tail in [
        b"\0\0\0\x20xml <x/>".as_slice(),
        b"\0\0\0\x10brobxml \xff\xff\xff\xff",
        b"\0\0\0\x07xml ",
    ] {
        let mut input = encoded.clone();
        input.extend_from_slice(tail);
        let (meta, pixels) = ok(
            json!({"operation":"decode","options":{"frame_index":0}}),
            &input,
        );
        assert_eq!(pixels, [7, 251]);
        assert_eq!(meta["complete"], false);
        let (code, meta, output) = call(json!({"operation":"decode_to_png"}), &input);
        assert_eq!(code, 1, "{meta}");
        assert!(output.is_empty());
    }
}

#[test]
fn metadata_free_png_keeps_output_icc_and_exact_samples() {
    for (dtype, input) in [("u8", vec![7, 251]), ("u16", vec![7, 0, 251, 255])] {
        let encoded = encode(spec(dtype, 1, input.len()), &input);
        let (info, _) = ok(json!({"operation":"info"}), &encoded);
        let (_, png) = ok(json!({"operation":"decode_to_png"}), &encoded);
        let reader = png::Decoder::new(std::io::Cursor::new(&png))
            .read_info()
            .unwrap();
        let profile = reader.info().icc_profile.as_ref().unwrap();
        assert_eq!(json!(profile.as_ref()), info["output_icc"]);
        let (meta, pixels) = ok(json!({"operation":"read_image"}), &png);
        assert_eq!(meta["frames"][0]["dtype"], dtype);
        assert_eq!(pixels, input);
    }
}
