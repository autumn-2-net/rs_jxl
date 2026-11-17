//! Frozen standard JXL Native reader-dispatch correctness and speed probes.
//! No Plus dependency. SAVE prepares fixtures; normal timing never encodes.
use rs_jxl::decode::{self, DecodeOptions};
use rs_jxl::entropy::{
    HybridUintConfig,
    options::{EntropyOptions, UintConfigStrategy},
    stream::Coding,
};
use rs_jxl::modular::{
    predictor::Predictor,
    tree::{Node, Tree},
};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode};
use std::{path::Path, time::Instant};
#[derive(Clone, Copy)]
struct Spec {
    name: &'static str,
    bits: u8,
    rgba: bool,
    predictor: Predictor,
    prefix: bool,
    non420: bool,
    lz: bool,
    ordinary: bool,
    fallback: bool,
    varying: bool,
}
fn specs() -> Vec<Spec> {
    let base = Spec {
        name: "wp8_ans_const",
        bits: 8,
        rgba: false,
        predictor: Predictor::Weighted,
        prefix: false,
        non420: false,
        lz: false,
        ordinary: false,
        fallback: false,
        varying: false,
    };
    vec![
        base,
        Spec {
            name: "wp8_offset_multiplier_fallback",
            fallback: true,
            ..base
        },
        Spec {
            name: "gradient8_offset_multiplier_fallback",
            fallback: true,
            predictor: Predictor::Gradient,
            ..base
        },
        Spec {
            name: "gradient8_ans_varying",
            varying: true,
            predictor: Predictor::Gradient,
            ..base
        },
        Spec {
            name: "gradient8_non420_varying",
            varying: true,
            non420: true,
            predictor: Predictor::Gradient,
            ..base
        },
        Spec {
            name: "wp16_prefix_const",
            bits: 16,
            prefix: true,
            ..base
        },
        Spec {
            name: "gradient31_ans_const",
            bits: 31,
            predictor: Predictor::Gradient,
            ..base
        },
        Spec {
            name: "wp8_non420",
            non420: true,
            ..base
        },
        Spec {
            name: "wp16_lz",
            bits: 16,
            lz: true,
            ..base
        },
        Spec {
            name: "gradient8_prefix_const",
            predictor: Predictor::Gradient,
            prefix: true,
            ..base
        },
        Spec {
            name: "rgba8_hidden",
            rgba: true,
            ordinary: true,
            ..base
        },
        Spec {
            name: "rgba16_hidden",
            rgba: true,
            bits: 16,
            ordinary: true,
            ..base
        },
        Spec {
            name: "rgba31_hidden",
            rgba: true,
            bits: 31,
            ordinary: true,
            ..base
        },
        Spec {
            name: "gray16_ordinary",
            bits: 16,
            ordinary: true,
            ..base
        },
    ]
}
fn image(spec: Spec, side: u32) -> Image {
    let channels = if spec.rgba { 4 } else { 1 };
    let max = (1u32 << spec.bits) - 1;
    let mut rng = 0x12345678u32;
    let values: Vec<u32> = (0..side as usize * side as usize * channels)
        .map(|i| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            if spec.lz {
                (i % 17) as u32
            } else if (!spec.ordinary && !spec.varying)
                || (spec.rgba && i % 4 == 3 && i / 4 % 3 == 0)
            {
                0
            } else {
                rng & max
            }
        })
        .collect();
    let data = if spec.bits <= 8 {
        PixelData::U8(values.iter().map(|&x| x as u8).collect())
    } else if spec.bits <= 16 {
        PixelData::U16(values.iter().map(|&x| x as u16).collect())
    } else {
        PixelData::U32(values)
    };
    Image::new(
        side,
        side,
        if spec.rgba {
            PixelLayout::Rgba
        } else {
            PixelLayout::Gray
        },
        data,
    )
    .unwrap()
}
fn options(spec: Spec) -> EncodeOptions {
    if spec.ordinary {
        return EncodeOptions {
            threads: 1,
            ..Default::default()
        };
    }
    EncodeOptions {
        threads: 1,
        modular: ModularOptions {
            rct: None,
            lz77: spec.lz,
            entropy: if spec.prefix {
                Coding::Prefix
            } else {
                Coding::Ans
            },
            custom_tree: Some(
                Tree::new(vec![Node::Leaf {
                    predictor: spec.predictor,
                    offset: if spec.fallback { -3 } else { 0 },
                    multiplier: if spec.fallback { 3 } else { 1 },
                }])
                .unwrap(),
            ),
            entropy_options: EntropyOptions {
                uint_config: UintConfigStrategy::Fixed(if spec.non420 {
                    HybridUintConfig::new(5, 1, 0).unwrap()
                } else {
                    HybridUintConfig::default()
                }),
                ..Default::default()
            },
            source_ma_defaults: false,
            source_tool_defaults: false,
            ..Default::default()
        },
        ..Default::default()
    }
}
fn words(data: &PixelData) -> Vec<u32> {
    match data {
        PixelData::U8(v) => v.iter().map(|&v| v as u32).collect(),
        PixelData::U16(v) => v.iter().map(|&v| v as u32).collect(),
        PixelData::U32(v) => v.clone(),
        _ => panic!("integer Native only"),
    }
}
fn verify(
    d: &decode::DecodedImage,
    width: u32,
    height: u32,
    bits: u8,
    channels: usize,
    expected: &[u32],
) {
    assert!(d.complete);
    assert_eq!(d.frames.len(), 1);
    let f = &d.frames[0];
    assert!(f.complete);
    assert_eq!(
        (f.width, f.height, f.layout.channels()),
        (width as usize, height as usize, channels)
    );
    assert_eq!(d.info.bit_depth.bits_per_sample(), bits as u32);
    assert_eq!(
        words(&f.pixels),
        expected,
        "all original words including invisible RGB"
    );
    let alpha = usize::from(channels == 4 || channels == 2);
    assert_eq!(f.extra_planes.len(), alpha);
    assert_eq!(d.info.extra_channels.len(), alpha);
    if alpha == 1 {
        let ec = &d.info.extra_channels[0];
        assert_eq!(ec.ec_type, decode::ExtraChannel::Alpha);
        assert_eq!(ec.bit_depth.bits_per_sample(), bits as u32);
        assert!(!ec.alpha_associated);
        assert_eq!(ec.dim_shift, 0);
        assert_eq!(
            words(&f.extra_planes[0]),
            expected
                .chunks_exact(channels)
                .map(|v| v[channels - 1])
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn reader_dispatch_native_integer_and_constant_residuals() {
    for spec in specs() {
        eprintln!("reader dispatch correctness {}", spec.name);
        let image = image(spec, 17);
        assert_eq!(image.bits(), spec.bits, "{} declared depth", spec.name);
        let bytes = encode(&image, &options(spec)).unwrap();
        let decoded = decode::decode(&bytes, &DecodeOptions::default()).unwrap();
        verify(
            &decoded,
            17,
            17,
            spec.bits,
            image.channel_count(),
            &words(image.pixels()),
        );
    }
}
fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |v, &b| {
        (v ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}
fn quoted(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn expected(bytes: &[u8]) -> (u32, u32, u8, usize, Vec<u32>) {
    assert!(bytes.len() >= 10);
    let width = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    let height = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let bits = bytes[8];
    let channels = bytes[9] as usize;
    assert_eq!(
        bytes.len(),
        10 + width as usize * height as usize * channels * 4
    );
    (
        width,
        height,
        bits,
        channels,
        bytes[10..]
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect(),
    )
}
fn real_cases() -> Vec<(String, String, String)> {
    let Some(path) = std::env::var_os("RS_JXL_READER_BENCH_REAL_TSV") else {
        return Vec::new();
    };
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|line| {
            let values: Vec<_> = line.split('\t').collect();
            assert_eq!(values.len(), 3);
            assert!(
                values[0]
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            );
            (
                values[0].to_owned(),
                values[1].to_owned(),
                values[2].to_owned(),
            )
        })
        .collect()
}
fn png_expected(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().unwrap();
    let mut storage = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut storage).unwrap();
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => panic!("expanded PNG required"),
    };
    let bits = match info.bit_depth {
        png::BitDepth::Eight => 8,
        png::BitDepth::Sixteen => 16,
        _ => panic!("8/16-bit PNG required"),
    };
    let mut raw = Vec::new();
    raw.extend(info.width.to_le_bytes());
    raw.extend(info.height.to_le_bytes());
    raw.extend([bits, channels]);
    let payload = &storage[..info.buffer_size()];
    if bits == 8 {
        for &v in payload {
            raw.extend(u32::from(v).to_le_bytes());
        }
    } else {
        for v in payload.chunks_exact(2) {
            raw.extend(u32::from(u16::from_be_bytes(v.try_into().unwrap())).to_le_bytes());
        }
    }
    raw
}
#[test]
#[ignore = "explicit frozen corpus directory; SAVE=1 prepares new inputs, otherwise decode only"]
fn native_reader_dispatch_speed() {
    let directory = std::env::var_os("RS_JXL_READER_BENCH_DIR").expect("RS_JXL_READER_BENCH_DIR");
    let root = Path::new(&directory);
    if std::env::var("SAVE").as_deref() == Ok("1") {
        assert!(!root.exists(), "SAVE requires new frozen corpus directory");
        std::fs::create_dir_all(root).unwrap();
        let mut manifest = Vec::new();
        for spec in specs() {
            let side = if spec.ordinary { 256 } else { 512 };
            let input = image(spec, side);
            let encoded = encode(&input, &options(spec)).unwrap();
            let source = words(input.pixels());
            verify(
                &decode::decode(&encoded, &DecodeOptions::default()).unwrap(),
                side,
                side,
                spec.bits,
                input.channel_count(),
                &source,
            );
            let mut raw = Vec::new();
            raw.extend(side.to_le_bytes());
            raw.extend(side.to_le_bytes());
            raw.extend([spec.bits, input.channel_count() as u8]);
            for word in source {
                raw.extend(word.to_le_bytes());
            }
            std::fs::write(root.join(format!("{}.jxl", spec.name)), &encoded).unwrap();
            std::fs::write(root.join(format!("{}.expected", spec.name)), &raw).unwrap();
            let identity = format!("{:016x} {:016x}", fingerprint(&encoded), fingerprint(&raw));
            std::fs::write(root.join(format!("{}.identity", spec.name)), identity).unwrap();
            manifest.push(format!("{{\"label\":{},\"jxl_bytes\":{},\"jxl_fnv64\":\"{:016x}\",\"expected_fnv64\":\"{:016x}\",\"width\":{side},\"height\":{side},\"bits\":{},\"channels\":{}}}",quoted(spec.name),encoded.len(),fingerprint(&encoded),fingerprint(&raw),spec.bits,input.channel_count()));
        }
        // These are existing standard streams and their original PNGs. Copy
        // once; never transcode/re-encode real controls in SAVE or timing mode.
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let jxls = repo.join("out/decoder-perf-20261024/pixel-inputs");
        let png_root = Path::new("D:/cpp_pj/jxl_python/jxl_testdata");
        for (label, jxl, png) in real_cases() {
            let png = if Path::new(&png).is_absolute() {
                Path::new(&png).to_path_buf()
            } else {
                png_root.join(&png)
            };
            let bytes = std::fs::read(jxls.join(&jxl)).unwrap();
            let raw = png_expected(&png);
            let (w, h, b, c, words) = expected(&raw);
            verify(
                &decode::decode(&bytes, &DecodeOptions::default()).unwrap(),
                w,
                h,
                b,
                c,
                &words,
            );
            std::fs::write(root.join(format!("{label}.jxl")), &bytes).unwrap();
            std::fs::write(root.join(format!("{label}.expected")), &raw).unwrap();
            std::fs::write(
                root.join(format!("{label}.identity")),
                format!("{:016x} {:016x}", fingerprint(&bytes), fingerprint(&raw)),
            )
            .unwrap();
            manifest.push(format!("{{\"label\":{},\"source_jxl\":{},\"source_png\":{},\"width\":{w},\"height\":{h},\"bits\":{b},\"channels\":{c},\"jxl_fnv64\":\"{:016x}\",\"expected_fnv64\":\"{:016x}\"}}",quoted(&label),quoted(&jxls.join(&jxl).to_string_lossy()),quoted(&png.to_string_lossy()),fingerprint(&bytes),fingerprint(&raw)));
        }
        let labels: Vec<_> = specs()
            .iter()
            .map(|s| s.name.to_owned())
            .chain(real_cases().into_iter().map(|s| s.0))
            .collect();
        std::fs::write(root.join("labels.txt"), labels.join("\n")).unwrap();
        std::fs::write(root.join("manifest.json"),format!("{{\"schema\":1,\"identity\":\"FNV64 noncryptographic frozen-file identity\",\"cases\":[{}]}}",manifest.join(","))).unwrap();
        return;
    }
    let mut rows = Vec::new();
    let labels = std::fs::read_to_string(root.join("labels.txt")).unwrap();
    for label in labels.lines() {
        let bytes = std::fs::read(root.join(format!("{label}.jxl"))).unwrap();
        let raw = std::fs::read(root.join(format!("{label}.expected"))).unwrap();
        let identity = std::fs::read_to_string(root.join(format!("{label}.identity"))).unwrap();
        assert_eq!(
            identity,
            format!("{:016x} {:016x}", fingerprint(&bytes), fingerprint(&raw))
        );
        let (width, height, bits, channels, values) = expected(&raw);
        let mut times = Vec::new();
        for round in 0..8 {
            let timer = Instant::now();
            let decoded = decode::decode(&bytes, &DecodeOptions::default()).unwrap();
            let seconds = timer.elapsed().as_secs_f64();
            verify(&decoded, width, height, bits, channels, &values);
            if round != 0 {
                times.push(seconds);
            }
        }
        let seconds = median(&mut times.clone());
        let mpix = width as f64 * height as f64 / 1e6;
        rows.push(format!("{{\"label\":{},\"jxl_bytes\":{},\"jxl_fnv64\":\"{:016x}\",\"expected_fnv64\":\"{:016x}\",\"width\":{width},\"height\":{height},\"bits\":{bits},\"channels\":{channels},\"seconds\":{:?},\"median_ms\":{},\"mpix_s\":{},\"all_native_samples_verified\":true}}",quoted(label),bytes.len(),fingerprint(&bytes),fingerprint(&raw),times,seconds*1000.0,mpix/seconds));
        eprintln!(
            "native reader {} {:.3} ms {:.3} MP/s",
            label,
            seconds * 1000.0,
            mpix / seconds
        );
    }
    let destination = std::env::var_os("RS_JXL_READER_BENCH_RECEIPT")
        .expect("RS_JXL_READER_BENCH_RECEIPT must distinguish baseline/after");
    std::fs::write(destination,format!("{{\"schema\":1,\"threads\":1,\"format\":\"Native integer\",\"api\":\"public decode::decode including allocations and per-call pool\",\"warmups\":1,\"repeats\":7,\"verification\":\"all words and authoritative alpha outside timer\",\"cases\":[{}]}}",rows.join(","))).unwrap();
}
