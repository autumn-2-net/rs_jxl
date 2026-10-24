//! Coefficient-only reconstruction regression coverage; no pixel re-encoding.
use rs_jxl::decode::{DecodeOptions, SampleFormat, reconstruct_jpeg};
use rs_jxl::jpeg::{TranscodeOptions, transcode};

fn minimal_jpeg() -> Vec<u8> {
    // One 8x8 grayscale block, unit quantization, zero DC and AC.
    let mut bytes = vec![0xff, 0xd8, 0xff, 0xdb, 0, 67, 0];
    bytes.extend([1; 64]);
    bytes.extend([0xff, 0xc0, 0, 11, 8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
    bytes.extend([0xff, 0xc4, 0, 38]);
    for selector in [0, 0x10] {
        bytes.extend([selector, 1]);
        bytes.extend([0; 15]);
        bytes.push(0);
    }
    bytes.extend([0xff, 0xda, 0, 8, 1, 1, 0, 0, 63, 0, 0x3f, 0xff, 0xd9]);
    bytes
}

#[test]
fn reconstruction_ignores_pixel_progression_and_format_options() {
    let original = minimal_jpeg();
    let encoded = transcode(&original, &TranscodeOptions::default()).unwrap();
    for threads in [1, 4] {
        let options = DecodeOptions {
            threads,
            format: SampleFormat::F32,
            include_preview: true,
            coalescing: false,
            allow_partial: true,
            downsampling: 8,
            ..Default::default()
        };
        assert_eq!(reconstruct_jpeg(&encoded, &options).unwrap(), original);
        let mut limited = options.clone();
        limited.limits.max_frames = 0;
        assert!(reconstruct_jpeg(&encoded, &limited).is_err());
        limited = options.clone();
        limited.limits.max_samples = 1;
        assert!(reconstruct_jpeg(&encoded, &limited).is_err());
        assert!(reconstruct_jpeg(&encoded[..encoded.len() / 2], &options).is_err());
    }
}

#[test]
#[ignore = "explicit JXL/JPEG path pairs; include independent djxl multipass/DC-precision fixtures"]
fn independent_recompression_forms_restore_original_bytes() {
    let paths = std::env::var_os("RS_JXL_JPEG_RECONSTRUCTION_PAIRS")
        .expect("set alternating JXL and original JPEG paths");
    let paths: Vec<_> = std::env::split_paths(&paths).collect();
    assert!(!paths.is_empty() && paths.len() <= 32 && paths.len().is_multiple_of(2));
    for pair in paths.chunks_exact(2) {
        for path in pair {
            assert!(std::fs::metadata(path).unwrap().len() <= 16 * 1024 * 1024);
        }
        let encoded = std::fs::read(&pair[0]).unwrap();
        let expected = std::fs::read(&pair[1]).unwrap();
        for threads in [1, 4] {
            let options = DecodeOptions {
                threads,
                ..Default::default()
            };
            assert_eq!(
                reconstruct_jpeg(&encoded, &options).unwrap(),
                expected,
                "{}",
                pair[0].display()
            );
        }
    }
}
