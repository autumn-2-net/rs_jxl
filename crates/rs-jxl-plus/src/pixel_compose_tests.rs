//! Pixel gate/LZ composition experiment; complete internal bytes, not PNG bytes.
use crate::{
    pixel_model::{self, Mode},
    pixel_tests::{samples, standard_samples},
};
use rs_jxl::{Image, PixelData, PixelLayout};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
const NAMES: [&str; 8] = [
    "plus_static",
    "plus_fitted_zero_gate",
    "plus_adaptive_zero_gate",
    "plus_static_lz",
    "plus_fitted_zero_gate_lz",
    "plus_adaptive_zero_gate_lz",
    "plus_selected",
    "standard_jxl_default",
];
fn mode(recipe: usize) -> Mode {
    match recipe {
        0 => Mode::Static,
        1 => Mode::Fitted,
        2 => Mode::Adaptive,
        3 => Mode::StaticLz,
        4 => Mode::FittedLz,
        5 => Mode::AdaptiveLz,
        6 => Mode::Selected,
        _ => unreachable!(),
    }
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
#[test]
#[ignore = "requires explicit integer PNG manifest and receipt destination"]
fn pixel_composition_corpus() {
    let manifest =
        std::env::var_os("RS_JXL_PLUS_PIXEL_MANIFEST").expect("RS_JXL_PLUS_PIXEL_MANIFEST");
    let destination = std::env::var_os("RS_JXL_PLUS_PIXEL_COMPOSE_RECEIPT")
        .expect("RS_JXL_PLUS_PIXEL_COMPOSE_RECEIPT");
    let destination_path = Path::new(&destination);
    if let Some(parent) = destination_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(destination_path)
        .expect("composition receipt must be a new file");
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
        let mut breakdowns = vec![Value::Null; 8];
        let mut selections = vec![Value::Null; 8];
        let mut lz_groups = vec![Value::Null; 8];
        for round in 0..4 {
            for position in 0..8 {
                let recipe = (position + round) % 8;
                let timer = Instant::now();
                let (encoded, stats) = if recipe < 7 {
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
                    selections[recipe] = json!(s.selected_groups);
                    lz_groups[recipe] = json!(s.lz_groups);
                    breakdowns[recipe] = json!({"groups":s.groups,"tree_bytes":s.tree_bytes,"wp_bytes":s.wp_bytes,"gate_bytes":s.gate_bytes,"table_bytes":s.table_bytes,"entropy_header_bytes":s.entropy_header_bytes,"entropy_data_bytes":s.entropy_data_bytes,"framing_bytes":framing});
                }
                let timer = Instant::now();
                enum Output {
                    Plus(pixel_model::Decoded),
                    Standard(rs_jxl::decode::DecodedImage),
                }
                let output = if recipe < 7 {
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
            json!({"recipe":name,"complete_bytes":size,"byte_breakdown":breakdowns[index],"selected_groups":selections[index],"lz_groups":lz_groups[index],"median_encode_seconds":median(runs.iter().map(|r|r["encode_seconds"].as_f64().unwrap()).collect()),"median_decode_seconds":median(runs.iter().map(|r|r["decode_seconds"].as_f64().unwrap()).collect())})
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
    let receipt = json!({"schema":1,"experiment":"pixel_lossless_composition","scope":"integer PNG samples, not PNG file byte reconstruction; JPEG results not extrapolated","strip_metadata":true,"color_conversion":false,"threads":1,"plus_tile_size":256,"plus_shared_tools":"same actual Modular MA training/transforms/HybridUint(4,2,0); gate and optional post-gate LZ differ","selected_search_candidates":["Static","Adaptive","StaticLz","AdaptiveLz"],"selected_encode_timing_includes_all_candidate_search":true,"standard":"complete default rs_jxl encoder, native vendored jxl-rs decoder; default decode cost includes duplicate native alpha extra plane, verified outside timer","warmup_rounds":1,"measured_rounds":3,"order":"recipes rotate by round","timing":"full encode including preparation and full decode; excludes source I/O, PNG input decoding and sample assertions; diagnostic local timing","source_identity":"per-case SHA256 copied from manifest; external verification performed separately","all_samples_include_invisible_rgb":true,"cases":rows,"totals":totals});
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(destination, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
}

#[test]
fn pixel_composition_modes_lz_and_mixed_groups_are_exact() {
    for image in [
        Image::new(
            2,
            1,
            PixelLayout::Rgba,
            PixelData::U8(vec![255, 7, 19, 0, 31, 91, 11, 255]),
        )
        .unwrap(),
        Image::new(
            2,
            1,
            PixelLayout::Rgb,
            PixelData::U32(vec![2147483647, 0, 17, 1, 2147483646, 99]),
        )
        .unwrap(),
        Image::new(
            2,
            1,
            PixelLayout::GrayAlpha,
            PixelData::U16(vec![65535, 0, 17, 65535]),
        )
        .unwrap(),
    ] {
        for recipe in 0..7 {
            let coded = pixel_model::encode(&image, mode(recipe)).unwrap();
            let decoded = pixel_model::decode(&coded.bytes).unwrap();
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
            assert_eq!(decoded.pixels, samples(image.pixels()));
        }
    }
    let values: Vec<u8> = (0..512usize * 3)
        .map(|i| {
            if i / 3 < 256 {
                ((i / 3) % 8) as u8
            } else {
                ((i * 197 + i / 7 * 31) % 256) as u8
            }
        })
        .collect();
    let image = Image::new(512, 1, PixelLayout::Rgb, PixelData::U8(values)).unwrap();
    let expected = samples(image.pixels());
    for recipe in 0..7 {
        let encoded = pixel_model::encode(&image, mode(recipe)).unwrap();
        assert_eq!(encoded.stats.groups, 2);
        assert_eq!(encoded.stats.selected_groups.iter().sum::<usize>(), 2);
        let decoded = pixel_model::decode(&encoded.bytes).unwrap();
        assert_eq!(
            (
                decoded.width,
                decoded.height,
                decoded.bits,
                decoded.channels
            ),
            (512, 1, 8, 3)
        );
        assert_eq!(decoded.pixels, expected);
    }
    // Compose two actual independently signaled group policies. The second
    // group's MA tree was trained with its true stream ID in the same image.
    let a = pixel_model::encode(&image, Mode::Static).unwrap();
    let b = pixel_model::encode(&image, Mode::Adaptive).unwrap();
    let first_end = 28 + 4 + u32::from_le_bytes(a.bytes[28..32].try_into().unwrap()) as usize;
    let second_begin = 28 + 4 + u32::from_le_bytes(b.bytes[28..32].try_into().unwrap()) as usize;
    let mut mixed = a.bytes[..first_end].to_vec();
    mixed.extend_from_slice(&b.bytes[second_begin..]);
    assert_eq!(pixel_model::decode(&mixed).unwrap().pixels, expected);
    // A repeated nonconstant tile creates profitable matches. All-zero input
    // is already a free singleton ANS symbol and need not select any LZ copy.
    let repeated = Image::new(
        64,
        64,
        PixelLayout::Gray,
        PixelData::U8(
            (0..64 * 64)
                .map(|i| {
                    let x = (i % 64) % 16;
                    let y = (i / 64) % 16;
                    ((x * 73 ^ y * 157 ^ x * y * 13) & 255) as u8
                })
                .collect(),
        ),
    )
    .unwrap();
    let wide = Image::new(
        64,
        64,
        PixelLayout::Gray,
        PixelData::U32(
            samples(repeated.pixels())
                .iter()
                .map(|&v| v * 8_388_583)
                .collect(),
        ),
    )
    .unwrap();
    for input in [&repeated, &wide] {
        for mode in [Mode::StaticLz, Mode::FittedLz, Mode::AdaptiveLz] {
            let coded = pixel_model::encode(input, mode).unwrap();
            assert!(
                coded.stats.lz_groups > 0,
                "synthetic must exercise real LZ matches"
            );
            assert_eq!(
                pixel_model::decode(&coded.bytes).unwrap().pixels,
                samples(input.pixels())
            );
        }
    }
}
#[test]
fn pixel_composition_rejects_reserved_mode_and_old_magic() {
    let image = Image::new(3, 2, PixelLayout::Rgba, PixelData::U8(vec![17; 24])).unwrap();
    let coded = pixel_model::encode(&image, Mode::Selected).unwrap();
    for (offset, value) in [(8usize, 1u8), (32, 3), (32, 8), (7, b'1')] {
        let mut bytes = coded.bytes.clone();
        bytes[offset] = value;
        assert!(
            pixel_model::decode(&bytes).is_err(),
            "invalid descriptor offset {offset}"
        );
    }
    assert!(pixel_model::decode(&coded.bytes[..coded.bytes.len() - 1]).is_err());
}
