//! Pixel-lossless dynamic entropy experiment; no PNG byte reconstruction claim.
use crate::pixel_model::{self, Mode};
use rs_jxl::{Image, PixelData, PixelLayout};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
const NAMES: [&str; 4] = [
    "plus_static",
    "plus_fitted_zero_gate",
    "plus_adaptive_zero_gate",
    "standard_jxl_default",
];
pub(super) fn samples(p: &PixelData) -> Vec<u32> {
    match p {
        PixelData::U8(v) => v.iter().map(|&x| u32::from(x)).collect(),
        PixelData::U16(v) => v.iter().map(|&x| u32::from(x)).collect(),
        PixelData::U32(v) => v.clone(),
        _ => panic!("integer samples only"),
    }
}
/// Default Native includes alpha in the main pixels AND as an authoritative
/// extra plane. Equal-depth input makes both exact; verify both, outside timing.
pub(super) fn standard_samples(
    image: &Image,
    d: &rs_jxl::decode::DecodedImage,
) -> (u32, u32, u8, usize, Vec<u32>) {
    assert!(d.complete);
    assert_eq!(d.frames.len(), 1);
    let f = &d.frames[0];
    assert!(f.complete);
    assert_eq!(f.layout, image.layout());
    let alpha = usize::from(image.layout().has_alpha());
    assert_eq!(d.info.extra_channels.len(), alpha);
    assert_eq!(f.extra_planes.len(), alpha);
    if alpha == 1 {
        let ec = &d.info.extra_channels[0];
        assert_eq!(ec.ec_type, rs_jxl::decode::ExtraChannel::Alpha);
        assert_eq!(ec.bit_depth.bits_per_sample(), u32::from(image.bits()));
        assert!(!ec.alpha_associated);
        assert_eq!(ec.dim_shift, 0);
        let original = samples(image.pixels());
        let expected: Vec<_> = original
            .chunks_exact(image.channel_count())
            .map(|p| p[image.channel_count() - 1])
            .collect();
        assert_eq!(
            samples(&f.extra_planes[0]),
            expected,
            "authoritative native alpha plane"
        );
    }
    (
        f.width as u32,
        f.height as u32,
        d.info.bit_depth.bits_per_sample() as u8,
        f.layout.channels(),
        samples(&f.pixels),
    )
}

#[test]
fn standard_native_alpha_duplicates_are_both_exact() {
    for image in [
        Image::new(
            3,
            2,
            PixelLayout::Rgba,
            PixelData::U8(vec![
                255, 7, 33, 0, 1, 2, 3, 255, 99, 8, 21, 0, 17, 91, 23, 1, 0, 0, 0, 0, 9, 8, 7, 254,
            ]),
        )
        .unwrap(),
        Image::new(
            3,
            2,
            PixelLayout::GrayAlpha,
            PixelData::U16(vec![65535, 0, 17, 65535, 32768, 0, 1, 1, 0, 65534, 999, 0]),
        )
        .unwrap(),
    ] {
        let encoded = rs_jxl::encode(
            &image,
            &rs_jxl::EncodeOptions {
                threads: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let decoded =
            rs_jxl::decode::decode(&encoded, &rs_jxl::decode::DecodeOptions::default()).unwrap();
        let (width, height, bits, channels, main) = standard_samples(&image, &decoded);
        assert_eq!(
            (width, height, bits, channels),
            (
                image.width(),
                image.height(),
                image.bits(),
                image.channel_count()
            )
        );
        assert_eq!(
            main,
            samples(image.pixels()),
            "main pixels retain hidden RGB and alpha"
        );
    }
}
fn mode(recipe: usize) -> Mode {
    match recipe {
        0 => Mode::Static,
        1 => Mode::Fitted,
        2 => Mode::Adaptive,
        _ => unreachable!(),
    }
}
fn check(image: &Image, decoded: &pixel_model::Decoded) {
    assert_eq!(
        (
            decoded.width,
            decoded.height,
            decoded.bits,
            decoded.channels
        ),
        (
            image.width(),
            image.height(),
            image.bits(),
            image.channel_count()
        )
    );
    assert_eq!(
        decoded.pixels,
        samples(image.pixels()),
        "all integer samples including invisible RGB"
    );
}
#[test]
fn pixel_integer_modes_exact() {
    let cases = vec![
        Image::new(
            3,
            2,
            PixelLayout::Rgba,
            PixelData::U8(vec![
                255, 3, 17, 0, 1, 222, 9, 255, 4, 5, 6, 0, 0, 0, 0, 0, 71, 9, 11, 1, 255, 255, 255,
                0,
            ]),
        )
        .unwrap(),
        Image::new(
            4,
            2,
            PixelLayout::Gray,
            PixelData::U16(vec![0, 65535, 1, 32768, 65534, 17, 99, 0]),
        )
        .unwrap(),
        Image::new(
            2,
            2,
            PixelLayout::Rgb,
            PixelData::U32(vec![
                0, 2147483647, 1, 2147483646, 65536, 17, 128, 0, 2147483647, 19, 99999999, 1,
            ]),
        )
        .unwrap(),
        Image::new(
            2,
            2,
            PixelLayout::GrayAlpha,
            PixelData::U16(vec![65535, 0, 17, 65535, 42, 0, 0, 1]),
        )
        .unwrap(),
        Image::new(3, 3, PixelLayout::Rgba, PixelData::U8(vec![0; 36])).unwrap(),
        Image::new(3, 3, PixelLayout::Rgb, PixelData::U8(vec![71; 27])).unwrap(),
    ];
    for image in cases {
        for recipe in 0..3 {
            let encoded = pixel_model::encode(&image, mode(recipe)).unwrap();
            check(&image, &pixel_model::decode(&encoded.bytes).unwrap());
        }
    }
}
#[test]
fn pixel_rejects_truncation_and_invalid_descriptor() {
    let image = Image::new(2, 2, PixelLayout::Rgb, PixelData::U8(vec![17; 12])).unwrap();
    for recipe in 0..3 {
        let encoded = pixel_model::encode(&image, mode(recipe)).unwrap();
        for n in [0, 1, encoded.bytes.len() / 2, encoded.bytes.len() - 1] {
            assert!(
                pixel_model::decode(&encoded.bytes[..n]).is_err(),
                "truncation {n}"
            );
        }
        let mut bad = encoded.bytes.clone();
        bad[0] ^= 255;
        assert!(pixel_model::decode(&bad).is_err());
        let mut bad = encoded.bytes;
        bad.push(0);
        assert!(pixel_model::decode(&bad).is_err());
    }
}

#[test]
fn pixel_caller_limits_and_partial_edge_tiles() {
    let (width, height) = (257u32, 259u32);
    let values: Vec<u8> = (0..width as usize * height as usize * 3)
        .map(|i| ((i / 3 * 17 + i % 3 * 71 + (i / 3 / width as usize) * 11) % 256) as u8)
        .collect();
    let sample_count = values.len();
    let image = Image::new(width, height, PixelLayout::Rgb, PixelData::U8(values)).unwrap();
    for recipe in 0..3 {
        let encoded = pixel_model::encode(&image, mode(recipe)).unwrap();
        assert!(
            pixel_model::decode_with_limits(&encoded.bytes, encoded.bytes.len() - 1, sample_count)
                .is_err()
        );
        assert!(
            pixel_model::decode_with_limits(&encoded.bytes, encoded.bytes.len(), sample_count - 1)
                .is_err()
        );
        check(
            &image,
            &pixel_model::decode_with_limits(&encoded.bytes, encoded.bytes.len(), sample_count)
                .unwrap(),
        );
    }
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
#[test]
#[ignore = "requires explicit integer PNG manifest and receipt destination"]
fn pixel_corpus() {
    let manifest =
        std::env::var_os("RS_JXL_PLUS_PIXEL_MANIFEST").expect("RS_JXL_PLUS_PIXEL_MANIFEST");
    let destination =
        std::env::var_os("RS_JXL_PLUS_PIXEL_RECEIPT").expect("RS_JXL_PLUS_PIXEL_RECEIPT");
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().expect("manifest cases");
    let standard_options = rs_jxl::EncodeOptions {
        threads: 1,
        ..rs_jxl::EncodeOptions::default()
    };
    let decode_options = rs_jxl::decode::DecodeOptions::default();
    let mut rows = Vec::new();
    for case in cases {
        let bytes = std::fs::read(case["source"].as_str().unwrap()).unwrap();
        assert!(
            bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
            "pixel manifest must contain PNG inputs"
        );
        let input = rs_jxl_image::decode(&bytes, true).unwrap();
        assert!(!input.jpeg);
        let image = input.image;
        let expected = samples(image.pixels());
        let mut repeats = Vec::new();
        let mut breakdowns = [Value::Null, Value::Null, Value::Null, Value::Null];
        for round in 0..4 {
            for position in 0..4 {
                let recipe = (position + round) % 4;
                let timer = Instant::now();
                let (encoded, stats) = if recipe < 3 {
                    let e = pixel_model::encode(&image, mode(recipe)).unwrap();
                    (e.bytes, Some(e.stats))
                } else {
                    (rs_jxl::encode(&image, &standard_options).unwrap(), None)
                };
                let encode_seconds = timer.elapsed().as_secs_f64();
                if round == 0
                    && let Some(s) = stats
                {
                    let sections = s.tree_bytes
                        + s.wp_bytes
                        + s.gate_bytes
                        + s.table_bytes
                        + s.entropy_header_bytes
                        + s.entropy_data_bytes;
                    let framing = encoded
                        .len()
                        .checked_sub(sections)
                        .expect("section costs exceed complete bytes");
                    breakdowns[recipe] = json!({"groups":s.groups,"tree_bytes":s.tree_bytes,"wp_bytes":s.wp_bytes,"gate_bytes":s.gate_bytes,"table_bytes":s.table_bytes,"entropy_header_bytes":s.entropy_header_bytes,"entropy_data_bytes":s.entropy_data_bytes,"framing_bytes":framing});
                }
                let timer = Instant::now();
                enum Output {
                    Plus(pixel_model::Decoded),
                    Standard(rs_jxl::decode::DecodedImage),
                }
                let output = if recipe < 3 {
                    Output::Plus(pixel_model::decode(&encoded).unwrap())
                } else {
                    Output::Standard(rs_jxl::decode::decode(&encoded, &decode_options).unwrap())
                };
                let decode_seconds = timer.elapsed().as_secs_f64();
                let (width, height, bits, channels, decoded) = match output {
                    Output::Plus(d) => (d.width, d.height, d.bits, d.channels, d.pixels),
                    Output::Standard(d) => standard_samples(&image, &d),
                };
                assert_eq!(
                    (width, height, bits, channels),
                    (
                        image.width(),
                        image.height(),
                        image.bits(),
                        image.channel_count()
                    )
                );
                assert_eq!(
                    decoded, expected,
                    "{} {} all original samples",
                    case["label"], NAMES[recipe]
                );
                if round > 0 {
                    repeats.push(json!({"round":round-1,"position":position,"recipe":NAMES[recipe],"complete_bytes":encoded.len(),"encode_seconds":encode_seconds,"decode_seconds":decode_seconds,"all_samples_verified":true}));
                }
            }
        }
        let variants:Vec<_>=NAMES.iter().enumerate().map(|(index,name)| {
            let runs:Vec<_>=repeats.iter().filter(|r|r["recipe"]==*name).collect();assert_eq!(runs.len(),3);
            let size=runs[0]["complete_bytes"].as_u64().unwrap();assert!(runs.iter().all(|r|r["complete_bytes"].as_u64()==Some(size)));
            json!({"recipe":name,"complete_bytes":size,"byte_breakdown":breakdowns[index],"median_encode_seconds":median(runs.iter().map(|r|r["encode_seconds"].as_f64().unwrap()).collect()),"median_decode_seconds":median(runs.iter().map(|r|r["decode_seconds"].as_f64().unwrap()).collect())})
        }).collect();
        rows.push(json!({"label":case["label"],"source":case["source"],"sha256":case["sha256"],"previously_used":case["previously_used"],"group":case["group"],"source_type":"PNG","original_png_bytes":bytes.len(),"width":image.width(),"height":image.height(),"pixels":u64::from(image.width())*u64::from(image.height()),"bits":image.bits(),"channels":image.channel_count(),"sample_storage":match image.pixels(){PixelData::U8(_)=>"u8",PixelData::U16(_)=>"u16",PixelData::U32(_)=>"u32",_=>unreachable!()},"variants":variants,"repeats":repeats}));
        eprintln!("pixel corpus {} complete", case["label"]);
        // A complete receipt after every image preserves useful progress if a
        // subsequent input fails. No automatic resume mixes experiment builds.
        write_receipt(Path::new(&destination), &rows);
    }
}
fn write_receipt(destination: &Path, rows: &[Value]) {
    let totals:Vec<_>=NAMES.iter().enumerate().map(|(index,name)|json!({"recipe":name,"complete_bytes":rows.iter().map(|r|r["variants"][index]["complete_bytes"].as_u64().unwrap()).sum::<u64>(),"sum_per_image_median_encode_seconds":rows.iter().map(|r|r["variants"][index]["median_encode_seconds"].as_f64().unwrap()).sum::<f64>(),"sum_per_image_median_decode_seconds":rows.iter().map(|r|r["variants"][index]["median_decode_seconds"].as_f64().unwrap()).sum::<f64>()})).collect();
    let receipt = json!({"schema":1,"experiment":"pixel_lossless_zero_gate","scope":"integer PNG samples, not PNG file byte reconstruction; JPEG results not extrapolated","strip_metadata":true,"color_conversion":false,"threads":1,"plus_tile_size":256,"plus_shared_tools":"same actual Modular MA training/transforms/HybridUint(4,2,0), no LZ; entropy gate differs","standard":"complete default rs_jxl encoder, native vendored jxl-rs decoder; default decode cost includes duplicate native alpha extra plane, verified outside timer","warmup_rounds":1,"measured_rounds":3,"order":"recipes rotate by round","timing":"full encode including preparation and full decode; excludes source I/O, PNG input decoding and sample assertions; diagnostic local timing","source_identity":"per-case SHA256 copied from manifest; external verification performed separately","all_samples_include_invisible_rgb":true,"cases":rows,"totals":totals});
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(destination, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
}
