//! Complete JPEG -> JXL -> original-JPEG checks, using the runtime Rust decoder.
//! No pixel re-encoding, original-JPEG lookup or external runtime decoder.
use rs_jxl::decode::{DecodeOptions, decode, reconstruct_jpeg};
use rs_jxl::jpeg::{TranscodeOptions, transcode};

#[test]
fn pixel_jxl_is_not_silently_reencoded_as_jpeg() {
    let image = rs_jxl::Image::new(
        2,
        2,
        rs_jxl::PixelLayout::Rgb,
        rs_jxl::PixelData::U8(vec![127; 12]),
    )
    .unwrap();
    let bytes = rs_jxl::encode(&image, &Default::default()).unwrap();
    assert!(reconstruct_jpeg(&bytes, &DecodeOptions::default()).is_err());
}

#[test]
fn reconstruction_rejects_empty_truncated_and_invalid_input() {
    for bytes in [&[][..], &[0xff, 0x0a][..], b"not JPEG XL"] {
        assert!(reconstruct_jpeg(bytes, &DecodeOptions::default()).is_err());
    }
}

#[test]
#[ignore = "explicit RS_JXL_JPEG_INPUTS path list; bounded real-data decoder acceptance"]
fn real_jpegs_restore_original_bytes_from_jxl_alone() {
    let roots = std::env::var_os("RS_JXL_JPEG_INPUTS").expect("set RS_JXL_JPEG_INPUTS");
    let mut paths = Vec::new();
    for path in std::env::split_paths(&roots) {
        if path.is_file() {
            paths.push(path);
        } else {
            for entry in std::fs::read_dir(path).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_file()
                    && entry.path().extension().is_some_and(|ext| {
                        ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg")
                    })
                {
                    paths.push(entry.path());
                }
                assert!(paths.len() <= 64, "select at most 64 JPEG fixtures");
            }
        }
    }
    paths.sort();
    paths.dedup();
    assert!(!paths.is_empty() && paths.len() <= 64);
    for path in paths {
        assert!(std::fs::metadata(&path).unwrap().len() <= 16 * 1024 * 1024);
        let source = std::fs::read(&path).unwrap();
        for effort in [3, 7] {
            let encoded = transcode(
                &source,
                &TranscodeOptions {
                    effort,
                    threads: 2,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("{} e{effort}: {error}", path.display()));
            for threads in [1, 4] {
                let options = DecodeOptions {
                    threads,
                    ..Default::default()
                };
                let restored = reconstruct_jpeg(&encoded, &options).unwrap_or_else(|error| {
                    panic!("{} e{effort} t{threads}: {error}", path.display())
                });
                assert_eq!(restored.len(), source.len(), "{}", path.display());
                assert!(
                    restored == source,
                    "{} e{effort} t{threads}: original bytes differ",
                    path.display()
                );
            }
            // Capture hooks must not make ordinary pixel decoding unusable.
            let pixels = decode(&encoded, &DecodeOptions::default()).unwrap();
            assert_eq!(pixels.frames.len(), 1);
            eprintln!(
                "{} e{effort}: {} bytes -> {} JXL bytes; JPEG exact at threads1/4; pixels decoded",
                path.file_name().unwrap().to_string_lossy(),
                source.len(),
                encoded.len()
            );
        }
    }
}
