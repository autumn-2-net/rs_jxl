//! Independent metadata inspection, not another pixel matrix. Parent runs this
//! ignored target with two caller-attested official tools in RS_JXL_JXLINFO.
//! Output contract: libjxl tools/jxlinfo.cc, with coalescing disabled and -v.

use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::entropy::stream::Coding;
use rs_jxl::frame::{BlendInfo, BlendMode, Frame, FrameOptions};
use rs_jxl::metadata::{Animation, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::multiframe::{self, Rect};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout};

const LOG_LIMIT: u64 = 64 << 10;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
static TEMP_ID: AtomicUsize = AtomicUsize::new(0);

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Self {
        let root = env::temp_dir().canonicalize().expect("resolve temp root");
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        for _ in 0..100 {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "rs-jxl-metadata-fields-{}-{timestamp}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        }
        panic!("could not create a unique metadata-fields test directory");
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            if thread::panicking() {
                eprintln!("cleanup {}: {error}", self.0.display());
            } else {
                panic!("cleanup {}: {error}", self.0.display());
            }
        }
    }
}

struct ReapOnDrop(Child);

impl Drop for ReapOnDrop {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

fn oracle_paths() -> Vec<PathBuf> {
    let value = env::var_os("RS_JXL_JXLINFO")
        .expect("RS_JXL_JXLINFO must list the two trusted 0.12/0.13 jxlinfo executables");
    let paths: Vec<_> = env::split_paths(&value)
        .map(|path| {
            assert!(path.is_file(), "invalid jxlinfo path: {}", path.display());
            path.canonicalize().expect("resolve jxlinfo executable")
        })
        .collect();
    assert_eq!(paths.len(), 2, "this gate requires both preserved oracles");
    assert_ne!(paths[0], paths[1], "duplicate oracle paths");
    // jxlinfo has no --version. The caller attests provenance using companion
    // cjxl/build records; neither a basename nor a directory label proves it.
    paths
}

fn read_log(path: &Path) -> String {
    let mut bytes = Vec::new();
    File::open(path)
        .unwrap()
        .take(LOG_LIMIT + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes.len() as u64 <= LOG_LIMIT, "oversized oracle log");
    String::from_utf8(bytes).expect("UTF-8 oracle log")
}

fn inspect(oracle: &Path, input: &Path, scratch: &ScratchDir) -> String {
    let stdout = scratch.0.join("stdout.log");
    let stderr = scratch.0.join("stderr.log");
    let mut command = Command::new(oracle);
    command
        .arg("-v")
        .arg(input)
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    let mut child = ReapOnDrop(
        command
            .spawn()
            .unwrap_or_else(|error| panic!("launch {}: {error}", oracle.display())),
    );
    drop(command); // Release parent redirection handles before Windows cleanup.
    let start = Instant::now();
    let status = loop {
        assert!(
            start.elapsed() < COMMAND_TIMEOUT,
            "{} exceeded {COMMAND_TIMEOUT:?}; child will be killed and reaped",
            oracle.display()
        );
        for log in [&stdout, &stderr] {
            assert!(
                fs::metadata(log).unwrap().len() <= LOG_LIMIT,
                "{} exceeded log budget",
                oracle.display()
            );
        }
        if let Some(status) = child.0.try_wait().expect("poll jxlinfo") {
            break status;
        }
        thread::sleep(Duration::from_millis(10));
    };
    drop(child);
    let out = read_log(&stdout);
    let err = read_log(&stderr);
    // jxlinfo can return success after BASIC_INFO followed by a decoder error.
    assert!(
        status.success() && err.is_empty(),
        "{}: {status}\nstdout:\n{out}\nstderr:\n{err}",
        oracle.display()
    );
    out
}

fn assert_fields(output: &str) {
    let lines: Vec<_> = output.lines().collect();
    for (prefix, expected) in [
        (
            "JPEG XL animation,",
            "JPEG XL animation, 4x3, (possibly) lossless, 8-bit RGB+Alpha+Depth",
        ),
        ("Number of color channels:", "Number of color channels: 3"),
        ("Number of extra channels:", "Number of extra channels: 2"),
        ("Have preview:", "Have preview: 0"),
        ("Have animation:", "Have animation: 1"),
        (
            "Ticks per second",
            "Ticks per second (numerator / denominator): 8 / 2",
        ),
        ("Num loops:", "Num loops: 3"),
        ("Have timecodes:", "Have timecodes: 1"),
        ("Intrinsic dimensions:", "Intrinsic dimensions: 4x3"),
        ("Orientation:", "Orientation: 1 (Normal)"),
        ("Animation length:", "Animation length: 2.250 seconds"),
    ] {
        let actual: Vec<_> = lines
            .iter()
            .copied()
            .filter(|line| line.starts_with(prefix))
            .collect();
        assert_eq!(actual, [expected], "field {prefix}\n{output}");
    }
    let extras: Vec<_> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with("Extra channel "))
        .map(|(index, line)| {
            std::iter::once(*line)
                .chain(
                    lines[index + 1..]
                        .iter()
                        .copied()
                        .take_while(|line| line.starts_with("  ")),
                )
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect();
    assert_eq!(
        extras,
        [
            "Extra channel 0:\n  type: Alpha\n  bits per sample: 8\n  name: opacity\n  alpha premultiplied: 0 (Non-premultiplied)",
            "Extra channel 1:\n  type: Depth\n  bits per sample: 12\n  upsampled 2x\n  name: depth-map",
        ],
        "extra descriptors\n{output}"
    );
    let frames: Vec<_> = lines
        .iter()
        .copied()
        .filter(|line| line.starts_with("Layer:") || line.starts_with("Frame:"))
        .collect();
    assert_eq!(
        frames,
        [
            "Layer: Full image size, duration: 0.0 ms, time code: 1020304, name: \"saved-base\"",
            "Frame: 2x2 at position (2,0), duration: 250.0 ms, time code: 1020305, name: \"replace-crop\"",
            "Frame: Full image size, duration: 500.0 ms, time code: 1020306, name: \"display-full\"",
        ],
        "complete ordered layer/frame list\n{output}"
    );
}

#[test]
#[ignore = "requires RS_JXL_JXLINFO OS path list of the two trusted official 0.12/0.13 tools"]
fn official_metadata_preserves_named_zero_duration_layer_crop_extras_and_animation() {
    let oracles = oracle_paths();
    let base = Image::new(
        4,
        3,
        PixelLayout::Rgba,
        PixelData::U8([7, 11, 13, 0].repeat(12)),
    )
    .unwrap();
    let mut metadata = base.metadata().clone();
    metadata.animation = Some(Animation {
        ticks_per_second_numerator: 8,
        ticks_per_second_denominator: 2,
        loops: 3,
        timecodes: true,
    });
    metadata.extra_channels[0].name = "opacity".into();
    let mut depth = ExtraChannelInfo::new(ExtraChannelType::Depth, 12).unwrap();
    depth.name = "depth-map".into();
    depth.dim_shift = 1;
    metadata.extra_channels.push(depth);
    let full = base
        .with_metadata(metadata, vec![PixelData::U16(vec![1, 4095, 17, 2049])])
        .unwrap();
    let crop = multiframe::crop(
        &full,
        Rect {
            x: 2,
            y: 0,
            width: 2,
            height: 2,
        },
    )
    .unwrap();
    let replace = BlendInfo {
        mode: BlendMode::Replace,
        source: 1,
        ..Default::default()
    };
    let frames = [
        Frame {
            image: &full,
            options: FrameOptions {
                name: "saved-base".into(),
                duration: 0,
                timecode: 0x0102_0304,
                save_as_reference: 1,
                // A regular zero-duration layer is exposed with coalescing=false.
                // A reference_only frame is not promised a JXL_DEC_FRAME event.
                ..Default::default()
            },
        },
        Frame {
            image: &crop,
            options: FrameOptions {
                name: "replace-crop".into(),
                origin: (2, 0),
                duration: 1,
                timecode: 0x0102_0305,
                blend: replace,
                extra_blend: vec![replace; 2],
                ..Default::default()
            },
        },
        Frame {
            image: &full,
            options: FrameOptions {
                name: "display-full".into(),
                duration: 2,
                timecode: 0x0102_0306,
                ..Default::default()
            },
        },
    ];
    let options = EncodeOptions {
        max_pixels: 12,
        max_output_bytes: 1 << 20,
        modular: ModularOptions {
            predictor: Predictor::Zero,
            rct: None,
            fixed_tree: Some(FixedTree::Zero),
            lz77: false,
            entropy: Coding::Prefix,
            ..Default::default()
        },
        ..Default::default()
    };
    let bytes = rs_jxl::encode_frames(&full, &frames, &options).unwrap();
    let scratch = ScratchDir::new();
    let input = scratch.0.join("metadata.jxl");
    fs::write(&input, bytes).unwrap();
    for oracle in oracles {
        eprintln!(
            "metadata oracle (caller-attested provenance): {}",
            oracle.display()
        );
        assert_fields(&inspect(&oracle, &input, &scratch));
    }
    // jxlinfo prints floating milliseconds, not raw ticks: these modest durations
    // are exactly representable, but u32 tick endpoints remain unproved. It does
    // not print save_as_reference, blend mode/source or per-extra blend headers.
}

#[test]
#[ignore = "requires RS_JXL_JXLINFO; source intrinsic display dimensions"]
fn official_metadata_preserves_intrinsic_size_without_resampling() {
    use rs_jxl::metadata::{IntrinsicSize, Orientation};
    let oracles = oracle_paths();
    let scratch = ScratchDir::new();
    let base = Image::new(4, 3, PixelLayout::Rgb, PixelData::U8([1, 2, 3].repeat(12))).unwrap();
    for size in [(8, 8), (24, 16), (13, 9), (513, 257), (262_145, 8193)] {
        for orientation in [Orientation::Identity, Orientation::Rotate90Clockwise] {
            let mut metadata = base.metadata().clone();
            metadata.intrinsic_size = Some(IntrinsicSize::new(size.0, size.1).unwrap());
            metadata.orientation = orientation;
            let image = base.clone().with_metadata(metadata, vec![]).unwrap();
            assert_eq!((image.width(), image.height()), (4, 3));
            let input = scratch.0.join("intrinsic.jxl");
            fs::write(
                &input,
                rs_jxl::encode(&image, &EncodeOptions::default()).unwrap(),
            )
            .unwrap();
            for oracle in &oracles {
                let output = inspect(oracle, &input, &scratch);
                assert!(
                    output
                        .lines()
                        .any(|line| line == format!("Intrinsic dimensions: {}x{}", size.0, size.1)),
                    "{output}"
                );
            }
        }
    }
}
