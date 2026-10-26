use super::*;

#[test]
fn cmyk_policy_is_explicit_and_does_not_enable_color_conversion_alone() {
    let default = args(&["in.jxl", "out.npy", "--color-space", "sRGB"]);
    let mut options = default.options(OutputFormat::Npy);
    default
        .configure_color(&mut options, OutputFormat::Npy)
        .unwrap();
    assert!(!options.color.cmyk_icc_conventions);
    let icc = args(&[
        "in.jxl",
        "out.npy",
        "--color-space",
        "sRGB",
        "--cmyk-icc-conventions",
    ]);
    icc.configure_color(&mut options, OutputFormat::Npy)
        .unwrap();
    assert!(options.color.cmyk_icc_conventions);
}
use std::{
    fs,
    sync::atomic::{AtomicU64, Ordering},
};

fn args(tail: &[&str]) -> DecodeArgs {
    let argv = ["rs-jxl", "decode"].into_iter().chain(tail.iter().copied());
    let crate::Command::Decode(args) = crate::Cli::try_parse_from(argv).unwrap().command else {
        panic!("wrong command")
    };
    args
}

#[test]
fn dither_defaults_and_exact_mode_names_parse() {
    use rs_jxl::decode::DitherMode;

    let default = args(&["in.jxl", "out.png"]);
    assert_eq!(default.options(OutputFormat::Png).dither, DitherMode::JxlRs);
    for (name, mode) in [
        ("jxl-rs", DitherMode::JxlRs),
        ("libjxl", DitherMode::Libjxl),
    ] {
        let a = args(&["in.jxl", "out.png", "--dither", name]);
        assert_eq!(a.options(OutputFormat::Png).dither, mode);
        // Output-only dithering must not reject byte reconstruction or metadata.
        for path in ["out.jpg", "out.xmp"] {
            let a = args(&["in.jxl", path, "--dither", name]);
            let format = a.output_format().unwrap();
            assert_eq!(a.options(format).dither, mode);
        }
        let a = args(&["in.jxl", "out.npy", "--stream-input", "--dither", name]);
        assert_eq!(a.options(OutputFormat::Npy).dither, mode);
    }
    for name in ["djxl", "jxlrs", "lib-jxl", "none", "JxlRs"] {
        assert!(
            crate::Cli::try_parse_from(
                ["rs-jxl", "decode", "in.jxl", "out.png", "--dither", name,]
            )
            .is_err()
        );
    }
}

#[test]
fn dither_does_not_change_native_output_or_metadata_sidecars() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    let source = rs_jxl::Image::new(
        3,
        1,
        PixelLayout::Gray,
        PixelData::U16(vec![0, 32960, 65535]),
    )
    .unwrap();
    fs::write(
        &input,
        rs_jxl::encode(&source, &Default::default()).unwrap(),
    )
    .unwrap();
    let mut results = Vec::new();
    for mode in ["jxl-rs", "libjxl"] {
        let output = dir.0.join(format!("{mode}.npy"));
        let metadata = dir.0.join(format!("{mode}.json"));
        let icc = dir.0.join(format!("{mode}.icc"));
        run(args(&[
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--dither",
            mode,
            "--metadata-out",
            metadata.to_str().unwrap(),
            "--orig-icc-out",
            icc.to_str().unwrap(),
            "--quiet",
        ]))
        .unwrap();
        results.push((
            fs::read(output).unwrap(),
            fs::read(metadata).unwrap(),
            fs::read(icc).unwrap(),
        ));
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn incremental_controls_are_explicit_and_bounded() {
    let a = args(&[
        "in.jxl",
        "out.npy",
        "--stream-input",
        "--input-chunk-size",
        "1",
    ]);
    assert!(a.stream_input);
    assert_eq!(a.input_chunk_size, Some(1));
    for tail in [
        vec!["in.jxl", "out.npy", "--input-chunk-size", "32"],
        vec![
            "in.jxl",
            "out.npy",
            "--stream-input",
            "--input-chunk-size",
            "0",
        ],
        vec![
            "in.jxl",
            "out.npy",
            "--stream-input",
            "--input-chunk-size",
            "1048577",
        ],
        vec!["in.jxl", "out.npy", "--stream-input", "--frame", "1"],
        vec!["in.jxl", "out.npy", "--stream-input", "--num-reps", "2"],
    ] {
        assert!(crate::Cli::try_parse_from(["rs-jxl", "decode"].into_iter().chain(tail)).is_err());
    }
}

#[test]
fn incremental_cli_matches_whole_input_without_publishing_truncation() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    let normal = dir.0.join("normal.npy");
    let streamed = dir.0.join("streamed.npy");
    let source =
        rs_jxl::Image::new(3, 2, PixelLayout::Rgb, PixelData::U8((0..18).collect())).unwrap();
    let encoded = rs_jxl::encode(&source, &Default::default()).unwrap();
    fs::write(&input, &encoded).unwrap();
    run(args(&[input.to_str().unwrap(), normal.to_str().unwrap()])).unwrap();
    run(args(&[
        input.to_str().unwrap(),
        streamed.to_str().unwrap(),
        "--stream-input",
        "--input-chunk-size",
        "1",
    ]))
    .unwrap();
    // Serialization identity is checked here, not compressed JXL identity.
    assert_eq!(fs::read(&normal).unwrap(), fs::read(&streamed).unwrap());
    fs::write(&input, &encoded[..encoded.len() / 2]).unwrap();
    let failed = dir.0.join("failed.npy");
    assert!(
        run(args(&[
            input.to_str().unwrap(),
            failed.to_str().unwrap(),
            "--stream-input"
        ]))
        .is_err()
    );
    assert!(!failed.exists());
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        for _ in 0..128 {
            let path = std::env::temp_dir().join(format!(
                "rs-jxl-decode-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            if fs::create_dir(&path).is_ok() {
                return Self(path);
            }
        }
        panic!("cannot create test directory")
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn djxl_aliases_custom_bits_and_benchmark_controls_parse() {
    let a = args(&[
        "in.jxl",
        "-",
        "--output_format",
        "PNM",
        "--bits_per_sample",
        "12",
        "--num_threads",
        "0",
        "--num_reps",
        "2",
        "--output_frames",
        "--output_extra_channels",
        "--norender_spotcolors",
        "--print_read_bytes",
        "--quiet",
        "-vv",
    ]);
    assert_eq!(a.output_format().unwrap(), OutputFormat::Pnm);
    let options = a.options(OutputFormat::Pnm);
    assert_eq!(options.format, SampleFormat::U16Bits(12));
    assert_eq!(options.threads, 1);
    assert!(!options.render_spot_colors);
    assert!(a.output_frames && a.output_extra_channels && a.print_read_bytes && a.quiet);
    assert_eq!((a.num_reps, a.verbose), (2, 2));
    for depth in ["0", "-1"] {
        let a = args(&["in.jxl", "out.pnm", "--bits_per_sample", depth]);
        assert_eq!(
            a.options(a.output_format().unwrap()).format,
            SampleFormat::Native
        );
    }
    let a = args(&["in.jxl", "--disable_output", "--num_threads=-1"]);
    assert!(a.output.is_none() && a.disable_output);
    assert!((1..=256).contains(&a.options(a.output_format().unwrap()).threads));
    assert_eq!(
        args(&["in.jxl", "--disable-output", "-J"])
            .output_format()
            .unwrap(),
        OutputFormat::Jpg
    );
}

#[test]
fn controls_reject_invalid_or_ineffective_requests() {
    for option in [
        vec!["--num_reps", "0"],
        vec!["--num_threads", "-2"],
        vec!["-s", "3"],
        vec!["--display_nits", "NaN"],
        vec!["--display_nits", "inf"],
        vec!["--display_nits=-1"],
        vec!["--background", "white"],
        vec!["--bits_per_sample", "31"],
        vec!["--frame", "0", "--output_frames"],
        vec!["--jpeg_quality", "90"],
        vec!["--pixels_to_jpeg"],
    ] {
        let argv = ["rs-jxl", "decode", "in.jxl", "out.png"]
            .into_iter()
            .chain(option);
        assert!(crate::Cli::try_parse_from(argv).is_err());
    }
    for tail in [
        vec!["in.jxl", "out.jpg", "--alpha_blend"],
        vec!["in.jxl", "out.png", "--no_coalescing"],
        vec!["in.jxl", "out.xmp", "--frame", "0"],
        vec!["in.jxl", "out.jpg", "--allow_partial_files"],
        vec!["in.jxl", "out.exr", "--bits", "16"],
    ] {
        assert!(args(&tail).output_format().is_err());
    }
    let mut a = args(&["in.jxl", "out.png", "--display_nits", "0"]);
    let mut options = a.options(OutputFormat::Png);
    a.configure_color(&mut options, OutputFormat::Png).unwrap();
    assert_eq!(options.color.display_nits, None);
    for n in [-1.0, f32::NAN, f32::INFINITY] {
        a.display_nits = Some(n);
        assert!(a.configure_color(&mut options, OutputFormat::Png).is_err());
    }
}

#[test]
fn frame_and_channel_filenames_preserve_parent_and_extension() {
    assert_eq!(
        outputs::numbered_path(
            Path::new("dir/out.ppm"),
            Some((3, 12)),
            Some(2),
            Some("pgm")
        )
        .unwrap(),
        Path::new("dir/out-03-ec2.pgm")
    );
    assert_eq!(
        outputs::numbered_path(Path::new("out.png"), None, None, None).unwrap(),
        Path::new("out.png")
    );
    assert_eq!(
        outputs::numbered_path(Path::new("-"), Some((0, 2)), Some(1), None).unwrap(),
        Path::new("-")
    );
}

#[test]
fn all_serialization_and_alias_preflight_precedes_publication() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    fs::write(&input, b"original").unwrap();
    let destination = dir.0.join("image.pgm");
    let a = args(&[
        input.to_str().unwrap(),
        destination.to_str().unwrap(),
        "--force",
    ]);
    let mut batch = outputs::Batch::new(&a);
    batch.add(&destination, b"first".to_vec(), true).unwrap();
    assert!(
        batch
            .add(
                &dir.0.join(".").join("image.pgm"),
                b"second".to_vec(),
                false
            )
            .is_err()
    );
    assert!(!destination.exists());
    assert_eq!(fs::read(&input).unwrap(), b"original");
    let mut batch = outputs::Batch::new(&a);
    batch.add(Path::new("-"), vec![1], true).unwrap();
    batch.add(Path::new("-"), vec![2], true).unwrap();
    assert!(batch.add(Path::new("-"), vec![3], false).is_err());
    assert_eq!(batch.remaining(), MAX_OUTPUT_BYTES - 2);
}

#[test]
fn separate_frames_and_planes_preserve_every_sample() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    fs::write(&input, b"original").unwrap();
    let path = dir.0.join("out.pgm");
    let mut f = frame(1, 1, PixelLayout::GrayAlpha, PixelData::U8(vec![17, 255]));
    f.extra_planes = vec![PixelData::U8(vec![255]), PixelData::U16(vec![0x1234])];
    let mut image = fixture(f.clone());
    image.info.extra_channels = vec![
        extra(ExtraChannel::Alpha, 8),
        extra(ExtraChannel::Depth, 16),
    ];
    f.index = 1;
    f.pixels = PixelData::U8(vec![19, 255]);
    image.frames.push(f);
    let a = args(&[
        input.to_str().unwrap(),
        path.to_str().unwrap(),
        "--output-frames",
        "--output-extra-channels",
    ]);
    let options = a.options(OutputFormat::Pgm);
    let mut batch = outputs::Batch::new(&a);
    outputs::pixels(&mut batch, &image, OutputFormat::Pgm, &options).unwrap();
    assert!(!dir.0.join("out-0.pgm").exists());
    batch.publish().unwrap();
    assert_eq!(
        fs::read(dir.0.join("out-0.pgm")).unwrap(),
        b"P5\n1 1\n255\n\x11"
    );
    assert_eq!(
        fs::read(dir.0.join("out-1.pgm")).unwrap(),
        b"P5\n1 1\n255\n\x13"
    );
    assert_eq!(
        fs::read(dir.0.join("out-0-ec0.pgm")).unwrap(),
        b"P5\n1 1\n255\n\xff"
    );
    assert_eq!(
        fs::read(dir.0.join("out-1-ec1.pgm")).unwrap(),
        b"P5\n1 1\n65535\n\x12\x34"
    );
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 7);
}

#[test]
fn output_frames_splits_coalesced_png_and_npy_with_invalid_apng_timebase() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    fs::write(&input, b"original").unwrap();
    let mut first = frame(1, 1, PixelLayout::Gray, PixelData::U8(vec![17]));
    first.duration_ticks = 1;
    let mut image = fixture(first.clone());
    first.index = 1;
    first.pixels = PixelData::U8(vec![19]);
    image.frames.push(first);
    image.info.animation = Some(decode::JxlAnimation {
        tps_numerator: 65537,
        tps_denominator: 1,
        num_loops: 0,
        have_timecodes: false,
    });
    assert!(serialize(&image, OutputFormat::Png, false, false).is_err());
    for (format, extension) in [(OutputFormat::Png, "png"), (OutputFormat::Npy, "npy")] {
        let path = dir.0.join(format!("out.{extension}"));
        let a = args(&[
            input.to_str().unwrap(),
            path.to_str().unwrap(),
            "--output-frames",
        ]);
        let options = a.options(format);
        assert!(options.coalescing);
        let mut batch = outputs::Batch::new(&a);
        outputs::pixels(&mut batch, &image, format, &options).unwrap();
        batch.publish().unwrap();
        assert!(!path.exists());
        for (index, sample) in [17, 19].into_iter().enumerate() {
            let bytes = fs::read(dir.0.join(format!("out-{index}.{extension}"))).unwrap();
            if format == OutputFormat::Png {
                let mut reader = png::Decoder::new(Cursor::new(bytes)).read_info().unwrap();
                assert!(reader.info().animation_control.is_none());
                let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
                let info = reader.next_frame(&mut pixels).unwrap();
                assert_eq!(&pixels[..info.buffer_size()], &[sample]);
                assert!(reader.next_frame(&mut pixels).is_err());
            } else {
                assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
                let start = 10 + usize::from(u16::from_le_bytes(bytes[8..10].try_into().unwrap()));
                assert!(
                    std::str::from_utf8(&bytes[10..start])
                        .unwrap()
                        .contains("'shape': (1, 1, 1)")
                );
                assert_eq!(&bytes[start..], &[sample]);
            }
        }
    }
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 5);
}

#[test]
fn rgba_alpha_blend_to_png_writes_rgb_and_can_preserve_alpha_sidecar() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    let source =
        rs_jxl::Image::new(1, 1, PixelLayout::Rgba, PixelData::U8(vec![255, 0, 0, 128])).unwrap();
    fs::write(
        &input,
        rs_jxl::encode(&source, &Default::default()).unwrap(),
    )
    .unwrap();
    for preserve in [false, true] {
        let path = dir
            .0
            .join(if preserve { "extras.png" } else { "plain.png" });
        let mut argv = vec![
            input.to_str().unwrap(),
            path.to_str().unwrap(),
            "--alpha_blend",
            "--quiet",
        ];
        if preserve {
            argv.push("--output_extra_channels");
        }
        run(args(&argv)).unwrap();
        let bytes = fs::read(path).unwrap();
        let mut png = png::Decoder::new(Cursor::new(bytes)).read_info().unwrap();
        assert_eq!(png.info().color_type, png::ColorType::Rgb);
        let mut pixels = vec![0; png.output_buffer_size().unwrap()];
        let info = png.next_frame(&mut pixels).unwrap();
        assert_eq!(info.buffer_size(), 3);
        assert_eq!(pixels[0], 255);
        assert!((127..=128).contains(&pixels[1]));
        if preserve {
            assert!(dir.0.join("extras-ec0.png").exists());
        }
    }
}

#[test]
fn disable_output_runs_repetitions_without_touching_image_or_sidecars() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    let source = rs_jxl::Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![17])).unwrap();
    fs::write(
        &input,
        rs_jxl::encode(&source, &Default::default()).unwrap(),
    )
    .unwrap();
    let output = dir.0.join("out.unsupported");
    let metadata = dir.0.join("meta.json");
    run(args(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--disable_output",
        "--num_reps",
        "2",
        "--metadata_out",
        metadata.to_str().unwrap(),
        "--quiet",
    ]))
    .unwrap();
    assert!(!output.exists() && !metadata.exists());
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
}
