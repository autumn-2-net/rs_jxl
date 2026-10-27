use super::args::parse_tps;
use super::input::{add_retained_bytes, compatible, load_images, retained_bytes, with_animation};
use super::report::{write_json_report, write_report};
use super::{EncodeFramesArgs, run};
use crate::{Cli, Command};
use clap::Parser;
use rs_jxl::metadata::Animation;
use rs_jxl::multiframe::{AutoEncodeResult, AutoOptions, Reference};
use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

fn parse(extra: &[&str]) -> EncodeFramesArgs {
    let mut argv = vec!["rs-jxl", "encode-frames", "b.png", "a.jpg", "-o", "out.jxl"];
    argv.extend_from_slice(extra);
    let Command::EncodeFrames(args) = Cli::try_parse_from(argv).unwrap().command else {
        panic!("expected encode-frames subcommand");
    };
    args
}

fn gray8() -> Image {
    Image::new(2, 1, PixelLayout::Gray, PixelData::U8(vec![0, 1])).unwrap()
}

#[test]
fn defaults_preserve_order_and_enable_safe_strategies() {
    let args = parse(&[]);
    assert_eq!(
        args.inputs,
        [PathBuf::from("b.png"), PathBuf::from("a.jpg")]
    );
    assert_eq!(args.frame_durations().unwrap(), [100, 100]);
    assert_eq!(args.animation(), Animation::default());
    assert_eq!(args.threads, 1);
    let auto = args.auto_options();
    assert_eq!(auto.reference, Reference::SmallestCrop);
    assert!(!auto.retained_reference && !args.report_json);
    assert!(!auto.reference_lookahead);
    assert_eq!(
        auto.reference_lookahead,
        AutoOptions::default().reference_lookahead
    );
    args.validate_options().unwrap();
    assert!(auto.allow_mask && auto.allow_multi_rect);
    assert_eq!(auto.prune, AutoOptions::default().prune);
    assert_eq!(
        auto.adaptive_rectangles,
        AutoOptions::default().adaptive_rectangles
    );
    assert_eq!(auto.tile_size, AutoOptions::default().tile_size);
    assert_eq!(
        auto.extra_tile_sizes,
        AutoOptions::default().extra_tile_sizes
    );
    assert_eq!(auto.max_rectangles, AutoOptions::default().max_rectangles);
    assert_eq!(auto.max_frames, usize::MAX);
    assert_eq!(auto.max_total_trials, usize::MAX);
    assert_eq!(auto.max_trial_pixels, u64::MAX);
}

#[test]
fn retained_reference_is_opt_in_and_requires_automatic_reference_selection() {
    for extra in [
        vec!["--retained-reference"],
        vec![
            "--retained-reference",
            "--reference",
            "auto",
            "--report-json",
        ],
    ] {
        let args = parse(&extra);
        args.validate_options().unwrap();
        assert!(args.auto_options().retained_reference);
    }
    assert!(parse(&["--report-json"]).report_json);
    for reference in ["previous", "first", "full"] {
        parse(&["--reference", reference])
            .validate_options()
            .unwrap();
        let args = parse(&["--reference", reference, "--retained-reference"]);
        assert!(
            args.validate_options()
                .unwrap_err()
                .to_string()
                .contains("--reference auto")
        );
        // Options must fail before nonexistent input files are accessed.
        assert!(
            run(args)
                .unwrap_err()
                .to_string()
                .contains("--reference auto")
        );
    }
}

#[test]
fn reference_lookahead_requires_retained_reference_before_io() {
    let error = Cli::try_parse_from([
        "rs-jxl",
        "encode-frames",
        "missing.png",
        "-o",
        "out.jxl",
        "--reference-lookahead",
    ])
    .unwrap_err();
    assert_eq!(
        error.kind(),
        clap::error::ErrorKind::MissingRequiredArgument
    );
    assert!(error.to_string().contains("--retained-reference"));
    let args = parse(&["--retained-reference", "--reference-lookahead"]);
    args.validate_options().unwrap();
    assert!(args.auto_options().reference_lookahead);
    assert!(args.auto_options().retained_reference);
    assert!(
        !parse(&["--retained-reference"])
            .auto_options()
            .reference_lookahead
    );
    for reference in ["previous", "first", "full"] {
        let args = parse(&[
            "--retained-reference",
            "--reference-lookahead",
            "--reference",
            reference,
        ]);
        assert!(
            run(args)
                .unwrap_err()
                .to_string()
                .contains("--reference auto")
        );
    }
    let mut args = parse(&[]);
    args.reference_lookahead = true;
    assert!(
        run(args)
            .unwrap_err()
            .to_string()
            .contains("--retained-reference")
    );
}

#[test]
fn default_frame_budget_accepts_more_than_ten_thousand_input_paths() {
    let mut args = parse(&[]);
    args.inputs = vec![PathBuf::from("frame.png"); 10_001];
    args.output = PathBuf::from("-");
    args.check_paths().unwrap();
    assert_eq!(args.frame_durations().unwrap().len(), 10_001);
    args.max_frames = 10_000;
    assert!(args.check_paths().is_err());
}

#[test]
fn rectangle_budget_accepts_values_above_the_former_4096_cap() {
    for limit in [4097usize, usize::MAX] {
        assert_eq!(
            parse(&["--max-rectangles", &limit.to_string()])
                .auto_options()
                .max_rectangles,
            limit
        );
    }
}

#[test]
fn explicit_timing_and_strategy_options_are_forwarded() {
    assert_eq!(
        parse(&["--extra-tile-sizes", "8,16,64"])
            .auto_options()
            .extra_tile_sizes,
        [8, 16, 64]
    );
    let args = parse(&[
        "--durations",
        "0,17",
        "--tps",
        "30000/1001",
        "--loops",
        "2",
        "--no-mask",
        "--no-multi-rect",
        "--no-prune",
        "--no-adaptive-rectangles",
        "--tile-size",
        "8",
        "--max-rectangles",
        "7",
        "--max-frames",
        "100",
        "--max-total-trials",
        "300",
        "--max-trial-pixels",
        "4000",
        "--threads",
        "2",
        "--rct",
        "0",
        "--predictor",
        "0",
    ]);
    assert_eq!(args.frame_durations().unwrap(), [0, 17]);
    assert_eq!(
        args.animation(),
        Animation {
            ticks_per_second_numerator: 30000,
            ticks_per_second_denominator: 1001,
            loops: 2,
            timecodes: false,
        }
    );
    let auto = args.auto_options();
    assert!(!auto.allow_mask && !auto.allow_multi_rect);
    assert!(!auto.prune && !auto.adaptive_rectangles);
    assert_eq!(
        (auto.tile_size, auto.max_rectangles, auto.max_frames),
        (8, 7, 100)
    );
    assert_eq!(auto.max_total_trials, 300);
    assert_eq!(auto.max_trial_pixels, 4000);
    let mut options = EncodeOptions::default();
    args.tools.apply(&mut options).unwrap();
    assert_eq!(options.modular.rct, Some(0));
    assert_eq!(args.threads, 2);
    for (name, reference) in [
        ("auto", Reference::SmallestCrop),
        ("previous", Reference::Previous),
        ("first", Reference::First),
        ("full", Reference::Full),
    ] {
        assert_eq!(
            parse(&["--reference", name]).auto_options().reference,
            reference
        );
    }
}

#[test]
fn pruning_and_adaptive_controls_are_independent_and_budgets_accept_zero() {
    let auto = parse(&["--no-prune"]).auto_options();
    assert!(!auto.prune && auto.adaptive_rectangles && auto.allow_multi_rect);
    let auto = parse(&["--no-adaptive-rectangles"]).auto_options();
    assert!(auto.prune && !auto.adaptive_rectangles && auto.allow_multi_rect);
    let args = parse(&[
        "--max-frames",
        "0",
        "--max-total-trials",
        "0",
        "--max-trial-pixels",
        "0",
    ]);
    let auto = args.auto_options();
    assert_eq!(auto.max_frames, 0);
    assert_eq!(auto.max_total_trials, 0);
    assert_eq!(auto.max_trial_pixels, 0);
    assert!(args.check_paths().is_err());
}

#[test]
fn durations_are_exact_and_never_implicitly_repeated_or_clamped() {
    assert_eq!(
        parse(&["--duration", "13"]).frame_durations().unwrap(),
        [13, 13]
    );
    assert_eq!(
        parse(&["--durations", "1", "--durations", "4294967295"])
            .frame_durations()
            .unwrap(),
        [1, u32::MAX]
    );
    assert!(parse(&["--durations", "1"]).frame_durations().is_err());
    assert!(parse(&["--durations", "1,2,3"]).frame_durations().is_err());
    assert!(
        Cli::try_parse_from([
            "rs-jxl",
            "encode-frames",
            "a",
            "-o",
            "b",
            "--duration",
            "1",
            "--durations",
            "2"
        ])
        .is_err()
    );
}

#[test]
fn invalid_timebases_limits_and_unsupported_search_fail() {
    for tps in [
        "0",
        "1/0",
        "1/1025",
        "1073741825",
        "1/2/3",
        "1/",
        "/2",
        "1.5",
        "x",
    ] {
        assert!(parse_tps(tps).is_err(), "{tps}");
    }
    assert!(parse_tps("1073741824/1024").is_ok());
    for extra in [
        vec!["--tile-size", "0"],
        vec!["--max-rectangles", "0"],
        vec!["--max-frames", "-1"],
        vec!["--max-total-trials", "-1"],
        vec!["--max-trial-pixels", "-1"],
        vec!["--threads", "257"],
        vec!["--duration", "4294967296"],
        vec!["--durations", "1,,2"],
        vec!["--reference", "unknown"],
        vec!["--effort", "3"],
        vec!["--source-effort", "3"],
        vec!["--source-search"],
    ] {
        let mut argv = vec!["rs-jxl", "encode-frames", "a", "-o", "b"];
        argv.extend(extra);
        assert!(Cli::try_parse_from(argv).is_err());
    }
    assert!(Cli::try_parse_from(["rs-jxl", "encode-frames", "-o", "b"]).is_err());
    assert!(Cli::try_parse_from(["rs-jxl", "encode-frames", "a"]).is_err());
    let mut args = parse(&[]);
    args.inputs = vec!["-".into(), "-".into()];
    assert!(
        args.check_paths()
            .unwrap_err()
            .to_string()
            .contains("at most once")
    );
    args.inputs = vec!["-".into()];
    args.output = "-".into();
    assert!(args.check_paths().is_ok());
    assert!(parse(&["--max-frames", "1"]).check_paths().is_err());
}

#[test]
fn compatibility_checks_geometry_storage_declared_depth_and_color() {
    let a = gray8();
    compatible(&a, &a).unwrap();
    let geometry = Image::new(1, 2, PixelLayout::Gray, PixelData::U8(vec![0, 1])).unwrap();
    assert!(compatible(&a, &geometry).is_err());
    let storage = Image::new(2, 1, PixelLayout::Gray, PixelData::U16(vec![0, 1]))
        .unwrap()
        .with_metadata(a.metadata().clone(), vec![])
        .unwrap();
    assert!(compatible(&a, &storage).is_err());
    let mut metadata = a.metadata().clone();
    metadata.bit_depth.bits_per_sample = 7;
    let declared = a.clone().with_metadata(metadata, vec![]).unwrap();
    assert!(compatible(&a, &declared).is_err());
    let mut metadata = a.metadata().clone();
    metadata.color_encoding =
        rs_jxl::color::ColorEncoding::linear_srgb(rs_jxl::color::ColorSpace::Gray);
    let color = a.clone().with_metadata(metadata, vec![]).unwrap();
    assert!(compatible(&a, &color).is_err());
    let alpha = Image::new(
        2,
        1,
        PixelLayout::GrayAlpha,
        PixelData::U8(vec![0, 0, 1, 1]),
    )
    .unwrap();
    assert!(compatible(&a, &alpha).is_err());
    let mut metadata = alpha.metadata().clone();
    metadata.extra_channels[0].bit_depth.bits_per_sample = 7;
    let declared_alpha = alpha.clone().with_metadata(metadata, vec![]).unwrap();
    assert!(compatible(&alpha, &declared_alpha).is_err());
    let mut metadata = alpha.metadata().clone();
    metadata.extra_channels[0].alpha_associated = true;
    let associated_alpha = alpha.clone().with_metadata(metadata, vec![]).unwrap();
    assert!(compatible(&alpha, &associated_alpha).is_err());
}

#[test]
fn animation_attachment_retains_samples_and_other_metadata() {
    let original = Image::new(
        1,
        1,
        PixelLayout::Rgba,
        PixelData::U16(vec![17, 1000, 65535, 0]),
    )
    .unwrap();
    let animation = parse(&["--tps", "60", "--loops", "3"]).animation();
    let animated = with_animation(original.clone(), animation).unwrap();
    assert_eq!(animated.pixels(), original.pixels());
    let mut metadata = original.metadata().clone();
    metadata.animation = Some(animation);
    assert_eq!(animated.metadata(), &metadata);
    assert!(with_animation(animated, animation).is_err());
    assert_eq!(retained_bytes(&original).unwrap(), 8);
    assert_eq!(
        add_retained_bytes(512 * 1024 * 1024, 8).unwrap(),
        512 * 1024 * 1024 + 8
    );
    assert_eq!(add_retained_bytes(usize::MAX - 8, 8).unwrap(), usize::MAX);
    assert!(add_retained_bytes(usize::MAX, 1).is_err());
}

#[test]
fn report_uses_actual_candidate_statistics_and_core_winner() {
    use rs_jxl::multiframe::{
        AutoCandidateReport, AutoDiffReport, AutoFrameReport, AutoPrunedReport, Rect, Strategy,
    };
    use std::time::Duration;

    let result = AutoEncodeResult {
        bytes: vec![0; 123],
        frames: vec![AutoFrameReport {
            winner_index: 1,
            retained_reference: true,
            reference_lookahead: true,
            lookahead_comparisons: 3,
            reused_payloads: 2,
            reused_payload_pixels: 25,
            references_before: [None, Some(0), Some(1), None],
            references_after: [None, Some(0), Some(1), None],
            candidates: vec![
                AutoCandidateReport {
                    strategy: Strategy::Full,
                    tile_size: None,
                    reference_index: None,
                    source: 0,
                    save_as_reference: 0,
                    encoded_bytes: 100,
                    encoded_frames: 1,
                    trial_pixels: 100,
                    elapsed: Duration::from_secs(1),
                },
                AutoCandidateReport {
                    strategy: Strategy::CropRetained,
                    tile_size: None,
                    reference_index: Some(1),
                    source: 2,
                    save_as_reference: 0,
                    encoded_bytes: 70,
                    encoded_frames: 2,
                    trial_pixels: 50,
                    elapsed: Duration::from_secs(3),
                },
            ],
            pruned: vec![AutoPrunedReport {
                strategy: Strategy::MergedRetained,
                tile_size: Some(16),
                reference_index: Some(1),
                source: 2,
                save_as_reference: 0,
                lower_bound: 80,
                encoded_frames: 1,
                planned_frames: 3,
                trial_pixels: 40,
                elapsed: Duration::from_millis(500),
            }],
            analysis_elapsed: Duration::from_millis(250),
            differences: vec![
                AutoDiffReport {
                    reference: 1,
                    source: 2,
                    changed_pixels: 0,
                    bounds: None,
                },
                AutoDiffReport {
                    reference: 0,
                    source: 1,
                    changed_pixels: 6,
                    bounds: Some(Rect {
                        x: 1,
                        y: 2,
                        width: 3,
                        height: 4,
                    }),
                },
            ],
        }],
    };
    let mut report = Vec::new();
    write_report(&mut report, &result).unwrap();
    let report = String::from_utf8(report).unwrap();
    assert_eq!(
        report,
        concat!(
            "frame 1: winner=1 candidates=2 pruned=1 analysis_elapsed=250ms retained_reference=true reference_lookahead=true lookahead_comparisons=3 reused_payloads=2 reused_payload_pixels=25 references_before=[None, Some(0), Some(1), None] references_after=[None, Some(0), Some(1), None]\n",
            "  difference reference=1 source=2 changed_pixels=0 bounds=None\n",
            "  difference reference=0 source=1 changed_pixels=6 bounds=Some(Rect { x: 1, y: 2, width: 3, height: 4 })\n",
            "  candidate=0 strategy=Full reference_index=None source=0 save_as_reference=0 encoded_bytes=100 encoded_frames=1 trial_pixels=100 elapsed=1s\n",
            "  candidate=1 strategy=CropRetained reference_index=Some(1) source=2 save_as_reference=0 encoded_bytes=70 encoded_frames=2 trial_pixels=50 elapsed=3s winner\n",
            "  pruned=0 strategy=MergedRetained reference_index=Some(1) source=2 save_as_reference=0 lower_bound=80 encoded_frames=1 planned_frames=3 trial_pixels=40 elapsed=500ms\n",
            "sequence: inputs=1 output_bytes=123\n",
        )
    );
    let mut json = Vec::new();
    write_json_report(&mut json, &result).unwrap();
    assert_eq!(json.iter().filter(|&&byte| byte == b'\n').count(), 1);
    assert_eq!(json.last(), Some(&b'\n'));
    let json: serde_json::Value = serde_json::from_slice(&json).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "operation": "encode_frames",
            "inputs": 1,
            "output_bytes": 123,
            "frames": [{
                "index": 0,
                "winner_index": 1,
                "retained_reference": true,
                "reference_lookahead": true,
                "lookahead_comparisons": 3,
                "reused_payloads": 2,
                "reused_payload_pixels": 25,
                "references_before": [null, 0, 1, null],
                "references_after": [null, 0, 1, null],
                "analysis_elapsed_seconds": 0.25,
                "candidates": [
                    {
                        "strategy": "Full",
                        "tile_size": null,
                        "reference_index": null,
                        "source": 0,
                        "save_as_reference": 0,
                        "encoded_bytes": 100,
                        "encoded_frames": 1,
                        "trial_pixels": 100,
                        "elapsed_seconds": 1.0,
                    },
                    {
                        "strategy": "CropRetained",
                        "tile_size": null,
                        "reference_index": 1,
                        "source": 2,
                        "save_as_reference": 0,
                        "encoded_bytes": 70,
                        "encoded_frames": 2,
                        "trial_pixels": 50,
                        "elapsed_seconds": 3.0,
                    },
                ],
                "pruned": [{
                    "strategy": "MergedRetained",
                    "tile_size": 16,
                    "reference_index": 1,
                    "source": 2,
                    "save_as_reference": 0,
                    "lower_bound": 80,
                    "encoded_frames": 1,
                    "planned_frames": 3,
                    "trial_pixels": 40,
                    "elapsed_seconds": 0.5,
                }],
                "differences": [
                    {
                        "reference": 1,
                        "source": 2,
                        "changed_pixels": 0,
                        "bounds": null,
                    },
                    {
                        "reference": 0,
                        "source": 1,
                        "changed_pixels": 6,
                        "bounds": {"x": 1, "y": 2, "width": 3, "height": 4},
                    },
                ],
            }],
        })
    );
}

struct TestDir(PathBuf);
impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        for _ in 0..128 {
            let path = std::env::temp_dir().join(format!(
                "rs-jxl-frames-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            if fs::create_dir(&path).is_ok() {
                return Self(path);
            }
        }
        panic!("cannot create test directory");
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn file_args(inputs: &[PathBuf], output: &Path, extra: &[&str]) -> EncodeFramesArgs {
    let mut argv = vec![OsString::from("rs-jxl"), OsString::from("encode-frames")];
    argv.extend(inputs.iter().map(|p| p.as_os_str().to_owned()));
    argv.push("--output".into());
    argv.push(output.as_os_str().to_owned());
    argv.extend(extra.iter().map(|arg| OsString::from(*arg)));
    let Command::EncodeFrames(args) = Cli::try_parse_from(argv).unwrap().command else {
        panic!("expected encode-frames");
    };
    args
}

fn write_png(path: &Path, depth: png::BitDepth, data: &[u8]) {
    let mut encoder = png::Encoder::new(fs::File::create(path).unwrap(), 2, 1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(depth);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(data).unwrap();
    writer.finish().unwrap();
}

#[test]
fn lookahead_png_sequence_roundtrips_with_mixed_zero_durations() {
    let dir = TestDir::new();
    let samples = [
        [17, 23, 31, 0, 42, 53, 64, 255],
        [91, 82, 73, 0, 64, 55, 46, 255],
        [92, 82, 73, 0, 64, 55, 46, 255],
        [18, 23, 31, 0, 42, 53, 64, 255],
    ];
    let inputs: Vec<_> = (0..samples.len())
        .map(|index| dir.0.join(format!("{index}.png")))
        .collect();
    for (path, data) in inputs.iter().zip(&samples) {
        write_png(path, png::BitDepth::Eight, data);
    }
    let output = dir.0.join("lookahead.jxl");
    run(file_args(
        &inputs,
        &output,
        &[
            "--retained-reference",
            "--reference-lookahead",
            "--report-json",
            "--durations",
            "0,3,0,5",
            "--threads",
            "2",
        ],
    ))
    .unwrap();
    let decoded = rs_jxl::decode::decode(&fs::read(&output).unwrap(), &Default::default()).unwrap();
    assert!(decoded.complete);
    assert_eq!(decoded.frames.len(), 2);
    for ((frame, index), duration) in decoded.frames.iter().zip([1, 3]).zip([3, 5]) {
        assert_eq!(frame.pixels, PixelData::U8(samples[index].to_vec()));
        assert_eq!(frame.duration_ticks, duration);
    }
}

#[test]
fn output_cannot_alias_any_input_even_with_force() {
    let dir = TestDir::new();
    let inputs = vec![dir.0.join("first.png"), dir.0.join("second.png")];
    for path in &inputs {
        fs::write(path, b"unchanged").unwrap();
    }
    for output in [&inputs[1], &dir.0.join(".").join("second.png")] {
        assert!(
            file_args(&inputs, output, &["--force"])
                .check_paths()
                .is_err()
        );
    }
    assert_eq!(fs::read(&inputs[1]).unwrap(), b"unchanged");
}

#[test]
fn failures_never_create_or_replace_output() {
    let dir = TestDir::new();
    let input = dir.0.join("frame.png");
    write_png(&input, png::BitDepth::Eight, &[1, 2, 3, 0, 4, 5, 6, 255]);
    let malformed = dir.0.join("bad.png");
    fs::write(&malformed, b"not an image").unwrap();
    let output = dir.0.join("out.jxl");
    let inputs = vec![input, malformed];
    assert!(run(file_args(&inputs, &output, &[])).is_err());
    assert!(!output.exists());
    fs::write(&output, b"existing output").unwrap();
    assert!(run(file_args(&inputs, &output, &["--force"])).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"existing output");
    assert!(
        run(file_args(
            &inputs,
            &output,
            &["--force", "--durations", "1"]
        ))
        .is_err()
    );
    assert_eq!(fs::read(&output).unwrap(), b"existing output");
}

#[test]
fn metadata_stripping_does_not_erase_color_mismatches() {
    let dir = TestDir::new();
    let inputs = vec![dir.0.join("srgb.png"), dir.0.join("linear.png")];
    let data = [1, 2, 3, 0, 4, 5, 6, 255];
    write_png(&inputs[0], png::BitDepth::Eight, &data);
    let mut encoder = png::Encoder::new(fs::File::create(&inputs[1]).unwrap(), 2, 1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::new(1.0));
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&data).unwrap();
    writer.finish().unwrap();
    let output = dir.0.join("out.jxl");
    for extra in [&[][..], &["--strip-metadata"][..]] {
        assert!(
            load_images(&file_args(&inputs, &output, extra))
                .unwrap_err()
                .to_string()
                .contains("identical metadata")
        );
    }
}

#[test]
fn unsupported_metadata_requires_explicit_stripping_without_sample_changes() {
    let dir = TestDir::new();
    let inputs = vec![dir.0.join("plain.png"), dir.0.join("comment.png")];
    let data = [1, 2, 3, 0, 4, 5, 6, 255];
    write_png(&inputs[0], png::BitDepth::Eight, &data);
    let mut encoder = png::Encoder::new(fs::File::create(&inputs[1]).unwrap(), 2, 1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&data).unwrap();
    writer
        .write_chunk(
            png::chunk::ChunkType(*b"tEXt"),
            b"Comment\0discard explicitly",
        )
        .unwrap();
    writer.finish().unwrap();
    let output = dir.0.join("out.jxl");
    assert!(
        load_images(&file_args(&inputs, &output, &[]))
            .unwrap_err()
            .to_string()
            .contains("--strip-metadata")
    );
    let images = load_images(&file_args(&inputs, &output, &["--strip-metadata"])).unwrap();
    assert_eq!(images[0].metadata(), images[1].metadata());
    for image in images {
        assert_eq!(image.pixels(), &PixelData::U8(data.to_vec()));
    }
}

#[test]
fn png_sequence_round_trip_preserves_hidden_rgb_depth_and_durations() {
    for depth in [png::BitDepth::Eight, png::BitDepth::Sixteen] {
        let dir = TestDir::new();
        let inputs = vec![
            dir.0.join("b.png"),
            dir.0.join("a.png"),
            dir.0.join("again.png"),
        ];
        let mut expected = Vec::new();
        for (index, path) in inputs.iter().enumerate() {
            let samples = [17 + u16::from(index == 1), 23, 31, 0, 42, 53, 64, 255];
            let (data, pixels) = if depth == png::BitDepth::Sixteen {
                let values: Vec<u16> = samples.iter().map(|v| v * 257).collect();
                (
                    values
                        .iter()
                        .flat_map(|v| v.to_be_bytes())
                        .collect::<Vec<_>>(),
                    PixelData::U16(values),
                )
            } else {
                let values: Vec<u8> = samples.iter().map(|v| *v as u8).collect();
                (values.clone(), PixelData::U8(values))
            };
            write_png(path, depth, &data);
            expected.push(pixels);
        }
        let output = dir.0.join("out.jxl");
        run(file_args(
            &inputs,
            &output,
            &[
                "--durations",
                "11,13,17",
                "--tps",
                "30000/1001",
                "--loops",
                "2",
                "--threads",
                if depth == png::BitDepth::Eight {
                    "1"
                } else {
                    "2"
                },
            ],
        ))
        .unwrap();
        let decoded =
            rs_jxl::decode::decode(&fs::read(&output).unwrap(), &Default::default()).unwrap();
        assert!(decoded.complete);
        assert_eq!(decoded.frames.len(), 3);
        let animation = decoded.info.animation.as_ref().unwrap();
        assert_eq!(
            (animation.tps_numerator, animation.tps_denominator),
            (30000, 1001)
        );
        assert_eq!(animation.num_loops, 2);
        for ((frame, pixels), duration) in decoded.frames.iter().zip(&expected).zip([11, 13, 17]) {
            assert_eq!(
                (frame.width, frame.height, frame.layout),
                (2, 1, PixelLayout::Rgba)
            );
            assert_eq!(&frame.pixels, pixels);
            assert_eq!(frame.duration_ticks, duration);
        }
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 4);
    }
}
