//! Native CLI smoke tests and opt-in real-image timing/oracle tests. The JPEG
//! reference is zune-jpeg's samples, never another decoder's IDCT/upsampling.

use std::fs::{self, File};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use rs_jxl::EncodeOptions;
use rs_jxl::scheduler::{
    CandidateParams, DriverOptions, NativeBackendOptions, native::encode_native_candidate,
    run_native_pixel_search,
};

use crate::{Cli, Result, encode_image, expect_encode, input, io};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn create() -> Result<Self> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        for _ in 0..128 {
            let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "rs-jxl-jpeg-oracle-{}-{timestamp}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err("could not create oracle test scratch directory".into())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn small_png(depth: png::BitDepth) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 8, 8);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(depth);
        let samples: Vec<u8> = (0..256)
            .flat_map(|i| {
                // Nonzero color under zero alpha must survive the input route.
                let value = if i % 4 == 3 { 0 } else { (i * 73) as u16 };
                if depth == png::BitDepth::Sixteen {
                    value.to_be_bytes().to_vec()
                } else {
                    vec![value as u8]
                }
            })
            .collect();
        encoder.write_header()?.write_image_data(&samples)?;
    }
    Ok(bytes)
}

#[test]
fn source_cli_small_png_matches_native_candidate_and_search() -> Result<()> {
    let scratch = Scratch::create()?;
    for depth in [png::BitDepth::Eight, png::BitDepth::Sixteen] {
        let png = small_png(depth)?;
        let input_path = scratch.0.join(format!("input-{depth:?}.png"));
        fs::write(&input_path, &png)?;
        let decoded = input::decode(&png, false)?;
        for search in [false, true] {
            let output_path = scratch.0.join(format!("output-{depth:?}-{search}.jxl"));
            let mut argv = vec![
                "rs-jxl".into(),
                "encode".into(),
                input_path.as_os_str().to_owned(),
                output_path.as_os_str().to_owned(),
                "--source-effort".into(),
                "3".into(),
                "--source-sample-limit".into(),
                "4096".into(),
            ];
            if search {
                argv.push("--source-search".into());
            }
            let args = expect_encode(Cli::try_parse_from(argv)?.command);
            crate::run(args)?;
            let mut base = EncodeOptions::default();
            base.modular.sample_limit = 4096;
            let params = CandidateParams::default().backend(3)?;
            let expected = if search {
                let result = run_native_pixel_search(
                    &decoded.image,
                    &params,
                    &DriverOptions::default(),
                    &base,
                    NativeBackendOptions::default(),
                )?;
                assert!(result.adaptations.iter().all(|a| a.gaps.is_empty()));
                assert!(
                    result
                        .adaptations
                        .iter()
                        .all(|a| a.resources.sample_limit == 4096)
                );
                result.output.bytes
            } else {
                let result = encode_native_candidate(
                    &params,
                    &decoded.image,
                    &base,
                    NativeBackendOptions::default(),
                )?;
                assert!(result.artifact.gaps.is_empty());
                assert_eq!(result.artifact.resources.sample_limit, 4096);
                result.bytes
            };
            assert!(!expected.is_empty());
            assert_eq!(
                fs::read(output_path)?,
                expected,
                "{depth:?}, search={search}"
            );
        }
    }
    Ok(())
}

#[test]
fn single_candidate_cli_accepts_all_source_efforts() -> Result<()> {
    let image = rs_jxl::Image::new(
        4,
        4,
        rs_jxl::PixelLayout::Gray,
        rs_jxl::PixelData::U8((0..16).map(|i| (i * 17) as u8).collect()),
    )?;
    for effort in 1..=10 {
        let args = expect_encode(
            Cli::try_parse_from([
                "rs-jxl",
                "encode",
                "-",
                "-",
                "--source-effort",
                &effort.to_string(),
            ])?
            .command,
        );
        let expected = encode_native_candidate(
            &CandidateParams::default().backend(effort)?,
            &image,
            &EncodeOptions::default(),
            NativeBackendOptions::default(),
        )?;
        assert_eq!(
            encode_image(&image, &args)?,
            expected.bytes,
            "effort {effort}"
        );
    }
    Ok(())
}

fn decode_with_deadline(mut command: Command) -> Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = command.spawn()?;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return if status.success() {
                    Ok(())
                } else {
                    Err(format!("official djxl failed with {status}").into())
                };
            }
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            result => {
                let _ = child.kill();
                let _ = child.wait();
                return match result {
                    Err(error) => Err(error.into()),
                    _ => Err("official djxl exceeded the 60 second timeout".into()),
                };
            }
        }
    }
}

#[test]
#[ignore = "requires RS_JXL_TEST_JPEG (real JPEG) and RS_JXL_DJXL (official decoder executable)"]
fn real_jpeg_pixels_match_official_djxl() -> Result<()> {
    let jpeg = PathBuf::from(
        std::env::var_os("RS_JXL_TEST_JPEG").ok_or("set RS_JXL_TEST_JPEG to a real JPEG file")?,
    );
    real_image_oracle(jpeg, false, true)
}

/// RS_JXL_SOURCE_EFFORT defaults to 3, RS_JXL_SOURCE_SEARCH to 0, and
/// RS_JXL_SOURCE_SAMPLE_LIMIT to 1048576. Times the CLI's encode dispatch only,
/// excluding input decode, filesystem I/O and the external reference decoder.
#[test]
#[ignore = "requires RS_JXL_TEST_IMAGE (real PNG/JPEG) and RS_JXL_DJXL (official decoder)"]
fn real_image_source_cli_pixels_and_performance() -> Result<()> {
    let path = PathBuf::from(
        std::env::var_os("RS_JXL_TEST_IMAGE").ok_or("set RS_JXL_TEST_IMAGE to a real PNG/JPEG")?,
    );
    real_image_oracle(path, true, false)
}

fn real_image_oracle(path: PathBuf, source: bool, require_jpeg: bool) -> Result<()> {
    let djxl =
        std::env::var_os("RS_JXL_DJXL").ok_or("set RS_JXL_DJXL to the official djxl executable")?;
    let bytes = io::read_input(&path)?;
    // The test opts into discarding metadata, not color or orientation changes.
    let reference = input::decode(&bytes, true)?;
    if require_jpeg {
        assert!(reference.jpeg, "RS_JXL_TEST_JPEG must contain a JPEG");
    }
    let scratch = Scratch::create()?;
    for threads in [1, 2] {
        let jxl = scratch.0.join(format!("encoded-{threads}.jxl"));
        let mut argv = vec![
            "rs-jxl".into(),
            "encode".into(),
            path.as_os_str().to_owned(),
            jxl.as_os_str().to_owned(),
            "--threads".into(),
            threads.to_string().into(),
            "--strip-metadata".into(),
        ];
        if source {
            argv.extend([
                "--source-effort".into(),
                std::env::var_os("RS_JXL_SOURCE_EFFORT").unwrap_or_else(|| "3".into()),
                "--source-sample-limit".into(),
                std::env::var_os("RS_JXL_SOURCE_SAMPLE_LIMIT").unwrap_or_else(|| "1048576".into()),
            ]);
            match std::env::var("RS_JXL_SOURCE_SEARCH").as_deref() {
                Ok("1") => argv.push("--source-search".into()),
                Ok("0") | Err(std::env::VarError::NotPresent) => {}
                _ => return Err("RS_JXL_SOURCE_SEARCH must be 0 or 1".into()),
            }
        }
        let args = expect_encode(Cli::try_parse_from(argv)?.command);
        let encode_started = Instant::now();
        let encoded = encode_image(&reference.image, &args)?;
        let encode_elapsed = encode_started.elapsed();
        let width = reference.image.width();
        let height = reference.image.height();
        let output_bytes = encoded.len();
        let pixels = u64::from(width) * u64::from(height);
        let bpp = output_bytes as f64 * 8.0 / pixels as f64;
        eprintln!(
            "input={} width={width} height={height} threads={threads} \
             source_effort={:?} source_search={} source_sample_limit={} \
             output_bytes={output_bytes} bpp={bpp:.6} pure_encode_ms={:.3}",
            path.display(),
            args.source_effort,
            args.source_search,
            args.source_sample_limit,
            encode_elapsed.as_secs_f64() * 1000.0,
        );
        let png = scratch.0.join(format!("decoded-{threads}.png"));
        let log = scratch.0.join(format!("djxl-{threads}.log"));
        io::write_output(&args.input, &args.output, args.force, &encoded)?;
        let mut command = Command::new(&djxl);
        let stderr = File::create(&log)?;
        command
            .arg(&jxl)
            .arg(&png)
            .arg("--num_threads=1")
            .stdin(Stdio::null())
            .stdout(Stdio::from(stderr.try_clone()?))
            .stderr(Stdio::from(stderr));
        if let Err(error) = decode_with_deadline(command) {
            return Err(
                format!("{error}\n{}", fs::read_to_string(&log).unwrap_or_default()).into(),
            );
        }
        let decoded = input::decode(&io::read_input(&png)?, true)?;
        assert!(!decoded.jpeg);
        assert_eq!(
            decoded.image.width(),
            reference.image.width(),
            "width, threads={threads}"
        );
        assert_eq!(
            decoded.image.height(),
            reference.image.height(),
            "height, threads={threads}"
        );
        assert_eq!(
            decoded.image.layout(),
            reference.image.layout(),
            "layout, threads={threads}"
        );
        assert_eq!(
            decoded.image.bits(),
            reference.image.bits(),
            "depth, threads={threads}"
        );
        assert_eq!(
            decoded.image.pixels(),
            reference.image.pixels(),
            "samples, threads={threads}"
        );
    }
    Ok(())
}
