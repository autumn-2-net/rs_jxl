// Requires the png 0.18 dev-dependency and official djxl executables only.
// Set RS_JXL_DJXL to an OS path list (semicolon-separated on Windows), then run:
// cargo test -p rs-jxl --test reference -- --ignored --nocapture
// Include both djxl 0.12 and 0.13 in the list to check both independent oracles.

use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout, encode};

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; native automatic streaming routing"]
fn native_streaming_boundary_matches_official_djxl() {
    use rs_jxl::scheduler::{
        native::{NativeBackendOptions, encode_native_candidate},
        params::{CandidateParams, Override},
    };
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let mut checks = 0;
    for bits in [8, 16] {
        for threads in [1, 4] {
            for (width, patches, streaming) in [
                (1024, Override::Default, false),
                (1025, Override::Default, true),
                (1025, Override::On, false),
            ] {
                let image = make_image(&Case {
                    width,
                    height: 3,
                    layout: PixelLayout::Rgba,
                    bits,
                    pattern: Pattern::Mixed,
                });
                let params = CandidateParams {
                    speed_tier: 1,
                    group_size_shift: Some(0),
                    patches,
                    ..Default::default()
                };
                let label = format!(
                    "native e9 width={width} bits={bits} threads={threads} patches={patches:?}"
                );
                let result = encode_native_candidate(
                    &params,
                    &image,
                    &EncodeOptions {
                        threads,
                        ..Default::default()
                    },
                    NativeBackendOptions::default(),
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(
                    result
                        .artifact
                        .options
                        .modular
                        .palette_policy
                        .as_ref()
                        .unwrap()
                        .groupwise,
                    streaming,
                    "{label}"
                );
                if streaming {
                    assert!(result.artifact.options.source_patches.is_none(), "{label}");
                }
                let input = scratch.0.join("native-streaming.jxl");
                fs::write(&input, result.bytes).unwrap();
                for decoder in &decoders {
                    let output = scratch.0.join("native-streaming.png");
                    run_checked(
                        Command::new(decoder)
                            .arg(&input)
                            .arg(&output)
                            .arg(format!("--bits_per_sample={bits}"))
                            .arg("--num_threads=1")
                            .arg("--quiet"),
                        &scratch.0,
                        &label,
                    );
                    assert_png_samples(&output, &image, &label);
                    fs::remove_file(output).unwrap();
                    checks += 1;
                }
            }
        }
    }
    eprintln!("{checks} native-streaming boundary pixel comparisons passed");
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; source channel/wave gather integration"]
fn parallel_source_gather_matches_official_djxl() {
    use rs_jxl::modular::sample::{GatherMode, GatherOptions, SamplingPolicy, SourcePolicy};
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let mut checks = 0;
    for bits in [8, 16] {
        let image = make_image(&Case {
            width: 513,
            height: 257,
            layout: PixelLayout::Rgba,
            bits,
            pattern: Pattern::Mixed,
        });
        for mode in [GatherMode::Channels, GatherMode::Waves] {
            for threads in [1, 4] {
                let mut options = EncodeOptions {
                    threads,
                    shared_tree: true,
                    ..Default::default()
                };
                options.modular.sampling = Some(
                    SamplingPolicy::from_source(&SourcePolicy {
                        speed_tier: 3,
                        learning_fraction: 0.2,
                        num_streams: 27,
                        ..Default::default()
                    })
                    .unwrap(),
                );
                options.modular.source_gather = Some(GatherOptions {
                    mode,
                    workers: 4,
                    batch_budget_bytes: if mode == GatherMode::Waves {
                        1536 << 10
                    } else {
                        16 << 20
                    },
                    ..GatherOptions::default()
                });
                options.modular.max_tree_leaves = rs_jxl::modular::ma_train::MAX_LEAVES;
                options.modular.source_tree_entropy = true;
                let label = format!("source gather {mode:?} threads={threads} depth={bits}");
                let encoded = encode(&image, &options).unwrap_or_else(|e| panic!("{label}: {e}"));
                let input = scratch.0.join("gather.jxl");
                fs::write(&input, encoded).unwrap();
                for decoder in &decoders {
                    let output = scratch.0.join("gather.png");
                    run_checked(
                        Command::new(decoder)
                            .arg(&input)
                            .arg(&output)
                            .arg(format!("--bits_per_sample={bits}"))
                            .arg("--num_threads=1")
                            .arg("--quiet"),
                        &scratch.0,
                        &label,
                    );
                    assert_png_samples(&output, &image, &label);
                    fs::remove_file(output).unwrap();
                    checks += 1;
                }
            }
        }
    }
    eprintln!("{checks} source-gather official pixel comparisons passed");
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; active sparse/stripe gathering"]
fn striped_sparse_source_gather_matches_official_djxl() {
    use rs_jxl::modular::sample::{GatherMode, GatherOptions, SamplingPolicy, SourcePolicy};
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for bits in [8, 16] {
        // A single 1024-wide group retains the source height>=512, area>=2^18
        // stripe gate; smaller groups would silently test unstriped gathering.
        let image = make_image(&Case {
            width: 513,
            height: 513,
            layout: PixelLayout::Rgb,
            bits,
            pattern: Pattern::Mixed,
        });
        for (mode, threads, stripes, sparse) in [
            (GatherMode::Channels, 1, 4, false),
            (GatherMode::Channels, 4, 4, true),
            (GatherMode::Waves, 4, 4, true),
        ] {
            let mut options = EncodeOptions {
                threads,
                shared_tree: true,
                group_size_shift: 3,
                ..Default::default()
            };
            options.modular.sampling = Some(
                SamplingPolicy::from_source(&SourcePolicy {
                    speed_tier: 3,
                    learning_fraction: 0.025,
                    num_streams: 1,
                    ..Default::default()
                })
                .unwrap(),
            );
            options.modular.source_gather = Some(GatherOptions {
                mode,
                workers: 4,
                batch_budget_bytes: 16 << 20,
                stripes,
                sparse,
            });
            options.modular.max_tree_leaves = rs_jxl::modular::ma_train::MAX_LEAVES;
            options.modular.source_tree_entropy = true;
            let label =
                format!("striped gather {mode:?} threads={threads} depth={bits} sparse={sparse}");
            let data = encode(&image, &options).unwrap_or_else(|e| panic!("{label}: {e}"));
            let input = scratch.0.join("stripes.jxl");
            fs::write(&input, data).unwrap();
            for decoder in &decoders {
                let output = scratch.0.join("stripes.png");
                run_checked(
                    Command::new(decoder)
                        .arg(&input)
                        .arg(&output)
                        .arg(format!("--bits_per_sample={bits}"))
                        .arg("--num_threads=1")
                        .arg("--quiet"),
                    &scratch.0,
                    &label,
                );
                assert_png_samples(&output, &image, &label);
                fs::remove_file(output).unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn grouped_ma_trees_match_official_djxl() {
    use rs_jxl::modular::{
        sample::{SamplingPolicy, SourcePolicy},
        tree_groups::TreeGroupOptions,
    };
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let mut checks = 0;
    for bits in [8, 16] {
        for shards in [2, 4, 8] {
            for responsive in [false, true] {
                let image = make_image(&Case {
                    width: 389,
                    height: 259,
                    layout: PixelLayout::Rgba,
                    bits,
                    pattern: Pattern::Mixed,
                });
                let mut options = EncodeOptions {
                    threads: 4,
                    shared_tree: true,
                    group_size_shift: 0,
                    responsive,
                    ..Default::default()
                };
                if responsive {
                    options.passes = rs_jxl::modular::layout::Passes::progressive();
                }
                options.modular.max_tree_leaves = rs_jxl::modular::ma_train::MAX_LEAVES;
                options.modular.sampling = Some(
                    SamplingPolicy::from_source(&SourcePolicy {
                        speed_tier: 3,
                        learning_fraction: 0.02,
                        ..Default::default()
                    })
                    .unwrap(),
                );
                options.modular.source_tree_entropy = true;
                options.modular.tree_groups = Some(TreeGroupOptions {
                    requested_shards: shards,
                    min_work: 1,
                    ..Default::default()
                });
                let context = format!("MA-shards={shards} bits={bits} responsive={responsive}");
                let bytes = encode(&image, &options).unwrap_or_else(|e| panic!("{context}: {e}"));
                let input = scratch.0.join("groups.jxl");
                fs::write(&input, bytes).unwrap();
                for decoder in &decoders {
                    let output = scratch.0.join("groups.png");
                    run_checked(
                        Command::new(decoder)
                            .arg(&input)
                            .arg(&output)
                            .arg(format!("--bits_per_sample={bits}"))
                            .arg("--num_threads=1")
                            .arg("--quiet"),
                        &scratch.0,
                        &context,
                    );
                    assert_png_samples(&output, &image, &context);
                    fs::remove_file(output).unwrap();
                    checks += 1;
                }
            }
        }
    }
    eprintln!("{checks} grouped-MA official pixel checks passed");
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn frame_area_streaming_matches_official_djxl() {
    check_frame_area_streaming(false);
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL; region input and TOC backpatch"]
fn region_input_seekable_output_matches_official_djxl() {
    check_frame_area_streaming(true);
}

fn check_frame_area_streaming(regions: bool) {
    use rs_jxl::{
        metadata::{ExtraChannelInfo, ExtraChannelType},
        modular::palette::PaletteOptions,
    };
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for bits in [8, 16] {
        for shift in [0u8, 1, 3] {
            let base = make_image(&Case {
                width: 1025,
                height: 257,
                layout: PixelLayout::Rgba,
                bits,
                pattern: Pattern::Mixed,
            });
            let mut metadata = base.metadata().clone();
            let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, bits).unwrap();
            extra.dim_shift = shift;
            metadata.extra_channels.push(extra);
            let count = 1025usize.div_ceil(1 << shift) * 257usize.div_ceil(1 << shift);
            let value = if bits == 8 { 37 } else { 37 * 257 };
            let plane = if bits == 8 {
                PixelData::U8(vec![value as u8; count])
            } else {
                PixelData::U16(vec![value; count])
            };
            let image = base.clone().with_metadata(metadata, vec![plane]).unwrap();
            let mut options = EncodeOptions {
                threads: 4,
                group_size_shift: 0,
                shared_tree: true,
                ..Default::default()
            };
            options.modular.palette_policy = Some(PaletteOptions {
                groupwise: true,
                palette_colors: 0,
                channel_colors_pre_transform_percent: 0.0,
                channel_colors_percent: 0.0,
                ..Default::default()
            });
            options.modular.source_tree_entropy = true;
            let context = format!("frame-area streaming 1025x257 bits={bits} extra_shift={shift}");
            let input = scratch.0.join("streaming.jxl");
            let encoded = if regions {
                use rs_jxl::{
                    region_input::ImageRegionSource, streaming::encode_regions_seekable_to,
                };
                let mut source = ImageRegionSource::new(&image).unwrap();
                let mut output = std::io::Cursor::new(Vec::new());
                let size = encode_regions_seekable_to(&mut source, &options, &mut output)
                    .unwrap_or_else(|e| panic!("{context}: {e}"));
                assert_eq!(output.position(), size as u64);
                assert_eq!(output.get_ref().len(), size);
                output.into_inner()
            } else {
                encode(&image, &options).unwrap_or_else(|e| panic!("{context}: {e}"))
            };
            fs::write(&input, encoded).unwrap();
            for decoder in &decoders {
                let output = scratch.0.join("streaming.png");
                run_checked(
                    Command::new(decoder)
                        .arg(&input)
                        .arg(&output)
                        .arg(format!("--bits_per_sample={bits}"))
                        .arg("--output_extra_channels")
                        .arg("--num_threads=1")
                        .arg("--quiet"),
                    &scratch.0,
                    &context,
                );
                assert_png_samples(&output, &base, &context);
                fs::remove_file(output).unwrap();
                let extra = [
                    scratch.0.join("streaming-ec1.png"),
                    scratch.0.join("streaming.png-ec1.png"),
                ]
                .into_iter()
                .find(|p| p.is_file())
                .unwrap_or_else(|| {
                    panic!(
                        "{context}: missing extra plane; files={:?}",
                        fs::read_dir(&scratch.0)
                            .unwrap()
                            .map(|p| p.unwrap().file_name())
                            .collect::<Vec<_>>()
                    )
                });
                let full_count = 1025 * 257;
                let plane = if bits == 8 {
                    PixelData::U8(vec![value as u8; full_count])
                } else {
                    PixelData::U16(vec![value; full_count])
                };
                let expected = Image::new(1025, 257, PixelLayout::Gray, plane).unwrap();
                assert_png_samples(&extra, &expected, &context);
                fs::remove_file(extra).unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn native_responsive_candidates_match_official_djxl() {
    use rs_jxl::scheduler::{
        CandidateParams, NativeBackendOptions, native::encode_native_candidate,
    };
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let mut checks = 0;
    for bits in [8, 16] {
        for layout in [PixelLayout::Gray, PixelLayout::Rgb, PixelLayout::Rgba] {
            let image = make_image(&Case {
                width: 137,
                height: 49,
                layout,
                bits,
                pattern: Pattern::Mixed,
            });
            for threads in [1, 4] {
                for speed_tier in [0, 1, 4, 7, 9] {
                    let params = CandidateParams {
                        responsive: true,
                        speed_tier,
                        group_size_shift: Some(0),
                        ..Default::default()
                    };
                    let label = format!(
                        "native responsive bits={bits} layout={layout:?} threads={threads} tier={speed_tier}"
                    );
                    let result = encode_native_candidate(
                        &params,
                        &image,
                        &EncodeOptions {
                            threads,
                            ..Default::default()
                        },
                        NativeBackendOptions::default(),
                    )
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert!(result.artifact.gaps.is_empty(), "{label}");
                    assert!(result.artifact.options.responsive, "{label}");
                    assert!(
                        result.artifact.options.modular.squeeze.is_empty(),
                        "{label}"
                    );
                    let input = scratch.0.join("responsive.jxl");
                    fs::write(&input, result.bytes).unwrap();
                    for decoder in &decoders {
                        let output = scratch.0.join("responsive.png");
                        run_checked(
                            Command::new(decoder)
                                .arg(&input)
                                .arg(&output)
                                .arg(format!("--bits_per_sample={bits}"))
                                .arg("--num_threads=1")
                                .arg("--quiet"),
                            &scratch.0,
                            &label,
                        );
                        assert_png_samples(&output, &image, &label);
                        fs::remove_file(output).unwrap();
                        checks += 1;
                    }
                }
            }
        }
    }
    eprintln!("{checks} native responsive pixel comparisons passed");
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn native_scheduler_retained_output_matches_official_djxl() {
    use rs_jxl::scheduler::{
        CandidateParams, DriverOptions, NativeBackendOptions, run_native_pixel_search,
    };
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for (bits, layout) in [(8, PixelLayout::Gray), (16, PixelLayout::Rgba)] {
        let image = make_image(&Case {
            width: 137,
            height: 49,
            layout,
            bits,
            pattern: Pattern::Mixed,
        });
        let base = EncodeOptions {
            threads: 2,
            ..Default::default()
        };
        let result = run_native_pixel_search(
            &image,
            &CandidateParams::default(),
            &DriverOptions::default(),
            &base,
            NativeBackendOptions::default(),
        )
        .unwrap();
        assert!(result.adaptations.iter().all(|a| a.gaps.is_empty()));
        assert!(
            result.output.report.full_encodes > 1,
            "must execute actual candidate search"
        );
        let record = &result.output.report.trials[result.output.report.winner_index];
        assert_eq!(record.bytes, Some(result.output.bytes.len()));
        let input = scratch.0.join("selected.jxl");
        fs::write(&input, result.output.bytes).unwrap();
        let context = format!("strict native search bits={bits} layout={layout:?}");
        for decoder in &decoders {
            let output = scratch.0.join("selected.png");
            run_checked(
                Command::new(decoder)
                    .arg(&input)
                    .arg(&output)
                    .arg(format!("--bits_per_sample={bits}"))
                    .arg("--num_threads=1")
                    .arg("--quiet"),
                &scratch.0,
                &context,
            );
            assert_png_samples(&output, &image, &context);
            fs::remove_file(output).unwrap();
        }
    }
}

const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Self {
        let root = env::temp_dir()
            .canonicalize()
            .expect("resolve temporary directory");
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        for _ in 0..100 {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "rs-jxl-reference-{}-{timestamp}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        }
        panic!("could not create a unique reference-test directory");
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

fn wait_with_timeout(command: &mut Command) -> io::Result<Option<ExitStatus>> {
    let mut child = ReapOnDrop(command.spawn()?);
    let start = Instant::now();
    loop {
        if let Some(status) = child.0.try_wait()? {
            return Ok(Some(status));
        }
        if start.elapsed() >= COMMAND_TIMEOUT {
            child.0.kill()?;
            child.0.wait()?;
            return Ok(None);
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn run_checked(command: &mut Command, scratch: &Path, context: &str) -> String {
    let stdout = scratch.join("stdout.log");
    let stderr = scratch.join("stderr.log");
    // Files avoid pipe backpressure while the parent polls the child deadline.
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    let result = wait_with_timeout(command);
    // Release the command's redirection handles before cleanup on Windows.
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let diagnostics = format!(
        "stdout:\n{}\nstderr:\n{}",
        fs::read_to_string(stdout).unwrap_or_default(),
        fs::read_to_string(stderr).unwrap_or_default()
    );
    match result {
        Ok(Some(status)) if status.success() => diagnostics,
        Ok(Some(status)) => panic!("{context}: djxl exited {status}\n{diagnostics}"),
        Ok(None) => {
            panic!("{context}: djxl exceeded {COMMAND_TIMEOUT:?}; killed and reaped\n{diagnostics}")
        }
        Err(error) => panic!("{context}: run djxl: {error}\n{diagnostics}"),
    }
}

fn decoder_paths() -> Vec<PathBuf> {
    let value = env::var_os("RS_JXL_DJXL").expect(
        "explicitly invoked reference test requires RS_JXL_DJXL: an OS path list of official djxl executables",
    );
    let paths: Vec<_> = env::split_paths(&value).collect();
    assert!(!paths.is_empty(), "RS_JXL_DJXL contains no decoder paths");
    paths
        .into_iter()
        .map(|path| {
            assert!(
                !path.as_os_str().is_empty() && path.is_file(),
                "RS_JXL_DJXL entry is empty or not a file: {}",
                path.display()
            );
            path.canonicalize().unwrap_or_else(|error| {
                panic!("resolve RS_JXL_DJXL entry {}: {error}", path.display())
            })
        })
        .collect()
}

fn assert_apng_samples(path: &Path, expected: &[Image]) {
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path).unwrap()));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    assert_eq!(
        reader.info().animation_control.as_ref().unwrap().num_frames as usize,
        expected.len()
    );
    for (index, image) in expected.iter().enumerate() {
        let mut bytes = vec![0; reader.output_buffer_size().unwrap()];
        let output = reader.next_frame(&mut bytes).unwrap();
        assert_eq!(
            (output.width, output.height),
            (image.width(), image.height()),
            "APNG frame {index}"
        );
        bytes.truncate(output.buffer_size());
        match image.pixels() {
            PixelData::U8(samples) => assert_eq!(&bytes, samples, "APNG frame {index}"),
            PixelData::U16(samples) => assert_eq!(
                bytes
                    .chunks_exact(2)
                    .map(|v| u16::from_be_bytes([v[0], v[1]]))
                    .collect::<Vec<_>>(),
                *samples,
                "APNG frame {index}"
            ),
            _ => panic!("PNG oracle only accepts U8/U16 fixtures"),
        }
    }
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL"]
fn layers_and_animation_match_official_djxl() {
    check_layers_and_animation(false, false);
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL; frame-aware source scheduler"]
fn searched_layers_and_animation_match_official_djxl() {
    check_layers_and_animation(true, false);
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL; incremental ordered jxlp output"]
fn streaming_layers_and_animation_match_official_djxl() {
    check_layers_and_animation(false, true);
}

#[test]
#[ignore = "requires official djxl; checks index offsets by independently decoding a suffix"]
fn indexed_streaming_seek_matches_official_djxl() {
    use rs_jxl::{
        frame::{Frame, FrameOptions},
        metadata::Animation,
    };
    let scratch = ScratchDir::new();
    let decoders = decoder_paths();
    for bits in [8, 16] {
        let images: Vec<_> = [Pattern::Mixed, Pattern::HiddenMaximum, Pattern::Mixed]
            .into_iter()
            .map(|pattern| {
                make_image(&Case {
                    width: 67,
                    height: 13,
                    layout: PixelLayout::Rgba,
                    bits,
                    pattern,
                })
            })
            .collect();
        let mut metadata = images[0].metadata().clone();
        metadata.animation = Some(Animation {
            ticks_per_second_numerator: 24,
            ..Default::default()
        });
        let canvas = images[0].clone().with_metadata(metadata, vec![]).unwrap();
        let frames: Vec<_> = images
            .iter()
            .zip([11, 22, 33])
            .map(|(image, duration)| Frame {
                image,
                options: FrameOptions {
                    duration,
                    ..Default::default()
                },
            })
            .collect();
        let mut output = Vec::new();
        rs_jxl::streaming::encode_indexed_frames_streaming_to(
            &canvas,
            &frames,
            &[0, 2],
            &EncodeOptions {
                threads: 4,
                ..Default::default()
            },
            &mut output,
        )
        .unwrap();
        let mut position = 0;
        let mut parts = Vec::new();
        let mut index = None;
        while position < output.len() {
            let length =
                u32::from_be_bytes(output[position..position + 4].try_into().unwrap()) as usize;
            assert!(length >= 8 && position + length <= output.len());
            match &output[position + 4..position + 8] {
                b"jxlp" => parts.push(&output[position + 12..position + length]),
                b"jxli" => {
                    assert!(index.is_none());
                    index = Some(&output[position + 8..position + length]);
                }
                _ => (),
            }
            position += length;
        }
        let payload = index.unwrap();
        let mut p = 0;
        fn varint(data: &[u8], p: &mut usize) -> u64 {
            let mut value = 0;
            for shift in (0..63).step_by(7) {
                let byte = data[*p];
                *p += 1;
                value |= u64::from(byte & 127) << shift;
                if byte & 128 == 0 {
                    return value;
                }
            }
            panic!("invalid index varint");
        }
        assert_eq!(varint(payload, &mut p), 2);
        assert_eq!(&payload[p..p + 8], &[0, 0, 0, 1, 0, 0, 0, 24]);
        p += 8;
        let first = [
            varint(payload, &mut p),
            varint(payload, &mut p),
            varint(payload, &mut p),
        ];
        let second = [
            varint(payload, &mut p),
            varint(payload, &mut p),
            varint(payload, &mut p),
        ];
        assert_eq!(p, payload.len());
        assert_eq!(parts.len(), 4);
        assert_eq!(first, [parts[0].len() as u64, 33, 2]);
        assert_eq!(second, [(parts[1].len() + parts[2].len()) as u64, 33, 1]);
        let stream = parts.concat();
        let mut suffix = parts[0].to_vec();
        suffix.extend_from_slice(&stream[(first[0] + second[0]) as usize..]);
        for (name, bytes, expected) in [
            ("full", output.as_slice(), images.as_slice()),
            ("seek", suffix.as_slice(), &images[2..]),
        ] {
            let input = scratch.0.join("index.jxl");
            fs::write(&input, bytes).unwrap();
            for decoder in &decoders {
                let decoded = scratch.0.join("index.png");
                run_checked(
                    Command::new(decoder)
                        .arg(&input)
                        .arg(&decoded)
                        .arg("--output_frames")
                        .arg(format!("--bits_per_sample={bits}"))
                        .arg("--num_threads=1")
                        .arg("--quiet"),
                    &scratch.0,
                    name,
                );
                assert_apng_samples(&decoded, expected);
                fs::remove_file(decoded).unwrap();
            }
        }
    }
}

fn check_layers_and_animation(search: bool, streaming: bool) {
    use rs_jxl::{
        encode_frames,
        frame::{BlendInfo, Frame, FrameOptions},
        metadata::Animation,
    };
    let scratch = ScratchDir::new();
    let decoders = decoder_paths();
    for bits in [8, 16] {
        let base = make_image(&Case {
            width: 67,
            height: 53,
            layout: PixelLayout::Rgba,
            bits,
            pattern: Pattern::Mixed,
        });
        let patch = make_image(&Case {
            width: 13,
            height: 17,
            layout: PixelLayout::Rgba,
            bits,
            pattern: Pattern::HiddenMaximum,
        });
        let replaced = match (base.pixels(), patch.pixels()) {
            (PixelData::U8(a), PixelData::U8(b)) => {
                let mut a = a.clone();
                for y in 0..17 {
                    a[((y + 9) * 67 + 11) * 4..((y + 9) * 67 + 24) * 4]
                        .copy_from_slice(&b[y * 13 * 4..(y + 1) * 13 * 4]);
                }
                PixelData::U8(a)
            }
            (PixelData::U16(a), PixelData::U16(b)) => {
                let mut a = a.clone();
                for y in 0..17 {
                    a[((y + 9) * 67 + 11) * 4..((y + 9) * 67 + 24) * 4]
                        .copy_from_slice(&b[y * 13 * 4..(y + 1) * 13 * 4]);
                }
                PixelData::U16(a)
            }
            _ => unreachable!(),
        };
        let expected = Image::new(67, 53, PixelLayout::Rgba, replaced).unwrap();
        for mode in ["animation", "layers", "reference"] {
            let mut metadata = base.metadata().clone();
            if mode == "animation" {
                metadata.animation = Some(Animation {
                    timecodes: true,
                    ..Default::default()
                });
            }
            let canvas = base.clone().with_metadata(metadata, vec![]).unwrap();
            let duration = u32::from(mode == "animation");
            let frames = [
                Frame {
                    image: &base,
                    options: FrameOptions {
                        duration,
                        save_as_reference: 1,
                        reference_only: mode == "reference",
                        ..Default::default()
                    },
                },
                Frame {
                    image: &patch,
                    options: FrameOptions {
                        origin: (11, 9),
                        duration,
                        timecode: if duration != 0 { 12345 } else { 0 },
                        blend: BlendInfo {
                            source: 1,
                            ..Default::default()
                        },
                        name: "exact RGBA crop".into(),
                        ..Default::default()
                    },
                },
            ];
            let options = EncodeOptions {
                shared_tree: true,
                threads: 2,
                ..Default::default()
            };
            let data = if streaming {
                let mut bytes = Vec::new();
                rs_jxl::streaming::encode_frames_streaming_to(
                    &canvas, &frames, &options, &mut bytes,
                )
                .unwrap();
                bytes
            } else if search {
                use rs_jxl::scheduler::{
                    CandidateParams, DriverOptions, NativeBackendOptions,
                    run_native_sequence_search,
                };
                let searched = run_native_sequence_search(
                    &canvas,
                    &frames,
                    &CandidateParams {
                        speed_tier: 7,
                        ..Default::default()
                    },
                    &DriverOptions::default(),
                    &options,
                    NativeBackendOptions::default(),
                )
                .unwrap();
                assert_eq!(searched.frames.len(), frames.len());
                for frame in &searched.frames {
                    assert!(frame.adaptations.iter().all(|a| a.gaps.is_empty()));
                    assert_eq!(
                        frame.report.trials[frame.report.winner_index].bytes,
                        Some(frame.encoded_bytes)
                    );
                }
                searched.bytes
            } else {
                encode_frames(&canvas, &frames, &options).unwrap()
            };
            let input = scratch.0.join("frames.jxl");
            fs::write(&input, data).unwrap();
            for decoder in &decoders {
                let output = scratch.0.join("frames.png");
                run_checked(
                    Command::new(decoder)
                        .arg(&input)
                        .arg(&output)
                        .arg("--output_frames")
                        .arg(format!("--bits_per_sample={bits}"))
                        .arg("--num_threads=1")
                        .arg("--quiet"),
                    &scratch.0,
                    mode,
                );
                if mode == "animation" {
                    assert_apng_samples(&output, &[base.clone(), expected.clone()]);
                } else {
                    assert_png_samples(&output, &expected, mode);
                }
                fs::remove_file(&output).unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL"]
fn automatic_frame_plans_match_official_djxl() {
    use rs_jxl::{
        encode_frames,
        metadata::Animation,
        multiframe::{self, Reference},
    };
    let scratch = ScratchDir::new();
    let decoders = decoder_paths();
    for bits in [8, 16] {
        for animated in [false, true] {
            let base = make_image(&Case {
                width: 71,
                height: 39,
                layout: PixelLayout::Rgba,
                bits,
                pattern: Pattern::Mixed,
            });
            let mut metadata = base.metadata().clone();
            if animated {
                metadata.animation = Some(Animation::default());
            }
            let base = base.with_metadata(metadata.clone(), vec![]).unwrap();
            let modified = match base.pixels() {
                PixelData::U8(v) => {
                    let mut v = v.clone();
                    v[5 * 4] = v[5 * 4].wrapping_add(1);
                    v[5 * 4 + 3] = 0;
                    v[(31 * 71 + 59) * 4 + 1] ^= 127;
                    PixelData::U8(v)
                }
                PixelData::U16(v) => {
                    let mut v = v.clone();
                    v[5 * 4] = v[5 * 4].wrapping_add(1);
                    v[5 * 4 + 3] = 0;
                    v[(31 * 71 + 59) * 4 + 1] ^= 32767;
                    PixelData::U16(v)
                }
                _ => unreachable!(),
            };
            let modified = Image::new(71, 39, PixelLayout::Rgba, modified)
                .unwrap()
                .with_metadata(metadata, vec![])
                .unwrap();
            let images = [base.clone(), modified.clone(), modified, base.clone(), base];
            for reference in [
                Reference::Full,
                Reference::Previous,
                Reference::First,
                Reference::SmallestCrop,
            ] {
                let plan = multiframe::plan(&images, &[u32::from(animated); 5], reference).unwrap();
                let frames: Vec<_> = plan.iter().map(|p| p.as_frame()).collect();
                let bytes = encode_frames(
                    &images[0],
                    &frames,
                    &EncodeOptions {
                        shared_tree: true,
                        threads: 2,
                        ..Default::default()
                    },
                )
                .unwrap();
                let input = scratch.0.join("planned.jxl");
                fs::write(&input, bytes).unwrap();
                let context = format!("planned {reference:?} {bits} bits animation={animated}");
                for decoder in &decoders {
                    let output = scratch.0.join("planned.png");
                    run_checked(
                        Command::new(decoder)
                            .arg(&input)
                            .arg(&output)
                            .arg("--output_frames")
                            .arg(format!("--bits_per_sample={bits}"))
                            .arg("--num_threads=1")
                            .arg("--quiet"),
                        &scratch.0,
                        &context,
                    );
                    if animated {
                        assert_apng_samples(&output, &images);
                    } else {
                        assert_png_samples(&output, images.last().unwrap(), &context);
                    }
                    fs::remove_file(output).unwrap();
                }
            }
        }
    }
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL"]
fn progressive_layout_matches_official_djxl() {
    use rs_jxl::modular::layout::Passes;
    let scratch = ScratchDir::new();
    let decoders = decoder_paths();
    let mut checks = 0;
    for (width, height) in [(19, 17), (259, 131), (1025, 17), (17, 1025)] {
        for bits in [8, 16] {
            for shared in [false, true] {
                for progressive in [false, true] {
                    let image = make_image(&Case {
                        width,
                        height,
                        layout: PixelLayout::Rgba,
                        bits,
                        pattern: Pattern::Mixed,
                    });
                    let options = EncodeOptions {
                        threads: 2,
                        shared_tree: shared,
                        responsive: true,
                        group_size_shift: 0,
                        passes: if progressive {
                            Passes::progressive()
                        } else {
                            Passes::default()
                        },
                        ..Default::default()
                    };
                    let context = format!(
                        "responsive {width}x{height} bits={bits} shared={shared} threepass={progressive}"
                    );
                    let bytes =
                        encode(&image, &options).unwrap_or_else(|e| panic!("{context}: {e}"));
                    let input = scratch.0.join("progressive.jxl");
                    fs::write(&input, bytes).unwrap();
                    for decoder in &decoders {
                        let output = scratch.0.join("progressive.png");
                        run_checked(
                            Command::new(decoder)
                                .arg(&input)
                                .arg(&output)
                                .arg(format!("--bits_per_sample={bits}"))
                                .arg("--num_threads=1")
                                .arg("--quiet"),
                            &scratch.0,
                            &context,
                        );
                        assert_png_samples(&output, &image, &context);
                        fs::remove_file(output).unwrap();
                        checks += 1;
                    }
                }
            }
        }
    }
    eprintln!("{checks} progressive layout pixel checks passed");
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL"]
fn source_tiers_match_official_djxl() {
    use rs_jxl::scheduler::{
        adapter::project_native,
        params::{CandidateParams, Override},
    };
    let scratch = ScratchDir::new();
    let decoders = decoder_paths();
    let mut checks = 0;
    for (width, height) in [(19, 17), (259, 19)] {
        for bits in [8, 16] {
            for tier in 0..=9 {
                let image = make_image(&Case {
                    width,
                    height,
                    layout: PixelLayout::Rgba,
                    bits,
                    pattern: Pattern::Mixed,
                });
                let params = CandidateParams {
                    speed_tier: tier,
                    patches: Override::Off,
                    ..Default::default()
                };
                let options = project_native(
                    &params,
                    &image,
                    &EncodeOptions {
                        threads: 2,
                        ..Default::default()
                    },
                )
                .unwrap()
                .exact_options()
                .unwrap();
                let context = format!("source tier={tier} {width}x{height} bits={bits}");
                let bytes = encode(&image, &options).unwrap_or_else(|e| panic!("{context}: {e}"));
                let input = scratch.0.join("source-tier.jxl");
                fs::write(&input, bytes).unwrap();
                for decoder in &decoders {
                    let output = scratch.0.join("source-tier.png");
                    run_checked(
                        Command::new(decoder)
                            .arg(&input)
                            .arg(&output)
                            .arg(format!("--bits_per_sample={bits}"))
                            .arg("--num_threads=1")
                            .arg("--quiet"),
                        &scratch.0,
                        &context,
                    );
                    assert_png_samples(&output, &image, &context);
                    fs::remove_file(output).unwrap();
                    checks += 1;
                }
            }
        }
    }
    eprintln!("{checks} native source-tier pixel checks passed");
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL"]
fn subsampled_extra_channels_match_official_djxl() {
    use rs_jxl::metadata::{ExtraChannelInfo, ExtraChannelType};
    let scratch = ScratchDir::new();
    let decoders = decoder_paths();
    for (width, height) in [(19, 17), (1025, 131)] {
        for shared in [false, true] {
            for responsive in [false, true] {
                let base = make_image(&Case {
                    width,
                    height,
                    layout: PixelLayout::Rgb,
                    bits: 16,
                    pattern: Pattern::Mixed,
                });
                let mut metadata = base.metadata().clone();
                let mut extras = Vec::new();
                for shift in 1..=3 {
                    let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
                    info.dim_shift = shift;
                    metadata.extra_channels.push(info);
                    let count = (width as usize).div_ceil(1 << shift)
                        * (height as usize).div_ceil(1 << shift);
                    extras.push(PixelData::U16(vec![1234 * u16::from(shift); count]));
                }
                let image = base.clone().with_metadata(metadata, extras).unwrap();
                let context = format!(
                    "shifted extras {width}x{height} shared={shared} responsive={responsive}"
                );
                let options = EncodeOptions {
                    threads: 2,
                    shared_tree: shared,
                    responsive,
                    group_size_shift: 0,
                    ..Default::default()
                };
                let bytes = encode(&image, &options).unwrap_or_else(|e| panic!("{context}: {e}"));
                let input = scratch.0.join("subsampled.jxl");
                fs::write(&input, bytes).unwrap();
                for decoder in &decoders {
                    let output = scratch.0.join("subsampled.png");
                    run_checked(
                        Command::new(decoder)
                            .arg(&input)
                            .arg(&output)
                            .arg("--bits_per_sample=16")
                            .arg("--output_extra_channels")
                            .arg("--num_threads=1")
                            .arg("--quiet"),
                        &scratch.0,
                        &context,
                    );
                    assert_png_samples(&output, &base, &context);
                    for i in 0..3 {
                        let layer = i + 1;
                        let path = [
                            scratch.0.join(format!("subsampled-ec{layer}.png")),
                            scratch.0.join(format!("subsampled.png-ec{layer}.png")),
                        ]
                        .into_iter()
                        .find(|p| p.is_file())
                        .expect("extra channel PNG");
                        // Official output is upsampled to canvas size. Constant planes
                        // verify framing/shift assignment without inventing its filter.
                        let expected = Image::new(
                            width,
                            height,
                            PixelLayout::Gray,
                            PixelData::U16(vec![
                                1234 * (i + 1) as u16;
                                width as usize * height as usize
                            ]),
                        )
                        .unwrap();
                        assert_png_samples(&path, &expected, &context);
                        fs::remove_file(path).unwrap();
                    }
                    fs::remove_file(output).unwrap();
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Mixed,
    Zero,
    Maximum,
    HiddenMaximum,
    Noise,
}

struct Case {
    width: u32,
    height: u32,
    layout: PixelLayout,
    bits: u8,
    pattern: Pattern,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for layout in [
        PixelLayout::Gray,
        PixelLayout::GrayAlpha,
        PixelLayout::Rgb,
        PixelLayout::Rgba,
    ] {
        for bits in [8, 16] {
            for (width, height) in [
                (1, 1),
                (1, 17),
                (17, 1),
                (2, 2),
                (17, 9),
                (256, 256),
                (257, 256),
                (2051, 17),
            ] {
                cases.push(Case {
                    width,
                    height,
                    layout,
                    bits,
                    pattern: Pattern::Mixed,
                });
            }
            // Singleton zero-bit prefix codes, both with and without group TOCs.
            for (width, height) in [(1, 1), (256, 256), (257, 256)] {
                cases.push(Case {
                    width,
                    height,
                    layout,
                    bits,
                    pattern: Pattern::Zero,
                });
            }
        }
    }
    // Exercise both axes of the 256-pixel group boundary without a large corpus.
    for (layout, bits) in [
        (PixelLayout::Gray, 8),
        (PixelLayout::GrayAlpha, 16),
        (PixelLayout::Rgb, 16),
        (PixelLayout::Rgba, 8),
    ] {
        cases.push(Case {
            width: 257,
            height: 259,
            layout,
            bits,
            pattern: Pattern::Mixed,
        });
    }
    // A second DC group (>2048 rows), with narrow rows to keep memory bounded.
    for (layout, bits) in [
        (PixelLayout::Gray, 16),
        (PixelLayout::GrayAlpha, 8),
        (PixelLayout::Rgb, 8),
        (PixelLayout::Rgba, 16),
    ] {
        cases.push(Case {
            width: 17,
            height: 2051,
            layout,
            bits,
            pattern: Pattern::Mixed,
        });
    }
    for bits in [8, 16] {
        for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
            cases.push(Case {
                width: 17,
                height: 9,
                layout,
                bits,
                pattern: Pattern::Maximum,
            });
        }
        cases.push(Case {
            width: 17,
            height: 9,
            layout: PixelLayout::Rgba,
            bits,
            pattern: Pattern::HiddenMaximum,
        });
    }
    cases.push(Case {
        width: 257,
        height: 259,
        layout: PixelLayout::Rgba,
        bits: 16,
        pattern: Pattern::Noise,
    });
    assert_eq!(cases.len(), 103);
    cases
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn parallel_channel_tokens_match_official_djxl() {
    use rs_jxl::modular::predictor::Predictor;
    use rs_jxl::modular::tree::Tree;
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for bits in [8, 16] {
        let image = make_image(&Case {
            width: 257,
            height: 65,
            layout: PixelLayout::Rgba,
            bits,
            pattern: Pattern::Mixed,
        });
        for threads in [1, 4] {
            let mut options = EncodeOptions {
                threads,
                shared_tree: true,
                group_size_shift: 3,
                ..Default::default()
            };
            options.modular.parallel_tokens = true;
            options.modular.previous_channels = 3;
            options.modular.custom_tree = Some(
                Tree::channel_predictors(&[
                    Predictor::Weighted,
                    Predictor::Gradient,
                    Predictor::Average0,
                    Predictor::Zero,
                ])
                .unwrap(),
            );
            let label = format!("token channels bits={bits} threads={threads}");
            let encoded =
                encode(&image, &options).unwrap_or_else(|error| panic!("{label}: {error}"));
            let input = scratch.0.join("token-channels.jxl");
            fs::write(&input, encoded).unwrap();
            for decoder in &decoders {
                let output = scratch.0.join("token-channels.png");
                run_checked(
                    Command::new(decoder)
                        .arg(&input)
                        .arg(&output)
                        .arg(format!("--bits_per_sample={bits}"))
                        .arg("--num_threads=1")
                        .arg("--quiet"),
                    &scratch.0,
                    &label,
                );
                assert_png_samples(&output, &image, &label);
                fs::remove_file(output).unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn source_lz77_index_block_and_suffix_tools_match_official_djxl() {
    check_source_lz_tools(0..17);
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL; source block DP variants"]
fn source_lz77_block_dp_variants_match_official_djxl() {
    check_source_lz_tools(17..26);
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL; generic fast/chain/interval routes"]
fn source_lz77_generic_methods_match_official_djxl() {
    check_source_lz_tools(26..43);
}

fn check_source_lz_tools(routes: std::ops::Range<usize>) {
    use rs_jxl::entropy::{HybridUintConfig, stream::Coding};
    use rs_jxl::modular::encode::FixedTree;
    use rs_jxl::tools::lz77::{
        SourceBlockDyadic, SourceBlockEnvelope, SourceBlockMode, SourceBucketMode,
        SourceChainCache, SourceConfig, SourceFastPrefix, SourceFastRanges, SourceIndexMode,
        SourceIndexTags, SourceIntervalDp, SourceLz77Method, SourceMatchCost, SourceParallelism,
        SourceSeedGuided, SourceSuffixMode,
    };
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for bits in [8, 16] {
        let image = make_image(&Case {
            width: 257,
            height: 65,
            layout: PixelLayout::Rgba,
            bits,
            pattern: Pattern::Mixed,
        });
        for route in routes.clone() {
            let mut source = SourceConfig {
                method: SourceLz77Method::Optc256,
                bucket_matcher: true,
                require_global_savings: false,
                ..Default::default()
            };
            match route {
                26..=37 => {
                    source.method = [
                        SourceLz77Method::Lz77b1w3f,
                        SourceLz77Method::Lz77b3w3f,
                        SourceLz77Method::Lz77b7w3f,
                        SourceLz77Method::Lz77b15w3f,
                        SourceLz77Method::Lz77b31w3f,
                        SourceLz77Method::Lz77b1w3t,
                        SourceLz77Method::Lz77b3w3t,
                        SourceLz77Method::Lz77b7w3t,
                        SourceLz77Method::Lz77b15w3t,
                        SourceLz77Method::Lz77b31w3t,
                        SourceLz77Method::Optc1,
                        SourceLz77Method::Optc3,
                    ][route - 26];
                    source.fast_ranges = SourceFastRanges::Both;
                    source.fast_prefix = Some(SourceFastPrefix::Simd);
                }
                38 => {
                    source.bucket_matcher = false;
                    source.chain_limit = Some(23);
                    source.chain_match_cache = SourceChainCache::Enabled;
                }
                39 => {
                    source.bucket_dominance = true;
                    source.match_cost = SourceMatchCost::SymbolId;
                }
                42 => {
                    source.method = SourceLz77Method::Lz77b31w3t;
                    source.fast_ranges = SourceFastRanges::Both;
                    source.fast_prefix = Some(SourceFastPrefix::Scalar);
                }
                0 | 17 => {
                    source.bucket_mode = SourceBucketMode::Short128;
                    source.bucket_distances = route == 17;
                }
                1 => source.bucket_mode = SourceBucketMode::PreciseRing,
                16 => {
                    source.compact_steps = true;
                    source.match_cost = SourceMatchCost::Frontier;
                }
                _ => {
                    source.match_cost = SourceMatchCost::Frontier;
                    source.index_mode = SourceIndexMode::All;
                    source.small_index = true;
                    source.index_tags = SourceIndexTags::Simd;
                    source.index_anchors_first = true;
                    source.parallel_index_build = true;
                    source.parallelism = Some(SourceParallelism::default());
                    source.block_mode = SourceBlockMode::Intervals;
                    match route {
                        2 => {
                            source.index_word_match = true;
                            source.index_match_cache = true;
                        }
                        3 => {
                            source.block_sparse_cache = true;
                            source.block_shared_frontier = true;
                            source.block_frontier_carry = true;
                            source.block_distance_bound = true;
                            source.block_distance_classes = true;
                        }
                        4 => {
                            source.shared_index = true;
                            source.block_distance_table = true;
                        }
                        5 => source.suffix_mode = SourceSuffixMode::Sliding,
                        6 => source.suffix_mode = SourceSuffixMode::SlidingLcp,
                        7 => source.suffix_mode = SourceSuffixMode::AdjacentLcp,
                        8 => source.block_seed_guided = SourceSeedGuided::Hints,
                        9 => source.block_seed_guided = SourceSeedGuided::SparseOrigins,
                        10 => {
                            source.block_seed_guided = SourceSeedGuided::Hints;
                            source.block_lazy = true;
                            source.suffix_mode = SourceSuffixMode::SlidingLcp;
                        }
                        11 | 12 => {
                            source.block_zero_tail = true;
                            source.block_literal_bound = true;
                            if route == 12 {
                                source.suffix_mode = SourceSuffixMode::AdjacentLcp;
                            }
                        }
                        13..=15 => {
                            source.index_compact_values = true;
                            source.index_word_match = true;
                            source.index_coordinates = route >= 14;
                            if route == 15 {
                                source.shared_index = true;
                            }
                        }
                        18 => source.block_calendar = true,
                        19 => source.block_vector_dp = true,
                        20 => source.block_range_dominance = true,
                        21 => source.block_dp_jump = true,
                        22 => source.block_dyadic = SourceBlockDyadic::Endpoints,
                        23 => source.block_dyadic = SourceBlockDyadic::Blocks16,
                        24 => source.block_envelope = SourceBlockEnvelope::Blocks64,
                        25 => {
                            source.block_envelope = SourceBlockEnvelope::Blocks16;
                            source.block_vector_dp = true;
                            source.block_range_dominance = true;
                            source.suffix_mode = SourceSuffixMode::SlidingLcp;
                        }
                        40 | 41 => {
                            source.block_mode = SourceBlockMode::Disabled;
                            source.index_interval_dp = if route == 40 {
                                SourceIntervalDp::Boundaries
                            } else {
                                SourceIntervalDp::AllEndpoints
                            };
                        }
                        _ => unreachable!(),
                    }
                }
            }
            let mut options = EncodeOptions {
                threads: 4,
                shared_tree: true,
                group_size_shift: 3,
                ..Default::default()
            };
            options.modular.fixed_tree = Some(FixedTree::GradientLeaf);
            options.modular.rct = None;
            options.modular.entropy = Coding::Ans;
            options.modular.entropy_options.lz77_length_config =
                HybridUintConfig::new(0, 0, 0).unwrap();
            options.modular.source_lz77 = Some(source);
            let label = format!("source LZ tool route={route} bits={bits}");
            let encoded =
                encode(&image, &options).unwrap_or_else(|error| panic!("{label}: {error}"));
            let input = scratch.0.join("source-lz-tools.jxl");
            fs::write(&input, encoded).unwrap();
            for decoder in &decoders {
                let output = scratch.0.join("source-lz-tools.png");
                run_checked(
                    Command::new(decoder)
                        .arg(&input)
                        .arg(&output)
                        .arg(format!("--bits_per_sample={bits}"))
                        .arg("--num_threads=1")
                        .arg("--quiet"),
                    &scratch.0,
                    &label,
                );
                assert_png_samples(&output, &image, &label);
                fs::remove_file(output).unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn source_ma_predictor_dedup_matches_official_djxl() {
    check_source_ma_tools(0);
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL; source MA histogram tools"]
fn source_ma_histogram_tools_match_official_djxl() {
    check_source_ma_tools(1);
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL; source MA entropy scoring tools"]
fn source_ma_scoring_tools_match_official_djxl() {
    for profile in 2..=4 {
        check_source_ma_tools(profile);
    }
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL; source MA quantile proposals"]
fn source_ma_quantile_search_matches_official_djxl() {
    for profile in 5..=7 {
        check_source_ma_tools(profile);
    }
}

#[test]
#[ignore = "requires official djxl via RS_JXL_DJXL; large MA race population"]
fn source_ma_statistical_race_matches_official_djxl() {
    use rs_jxl::modular::{
        ma_train::SearchOptions,
        predictor::Predictor,
        sample::{SamplingPolicy, SourcePolicy},
    };
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let mut state = 0x1e68_540bu32;
    let samples = (0..1024 * 512)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u16
        })
        .collect();
    let image = Image::new(1024, 512, PixelLayout::Gray, PixelData::U16(samples)).unwrap();
    let mut options = EncodeOptions {
        threads: 4,
        shared_tree: true,
        ..Default::default()
    };
    options.modular.predictor = Predictor::Variable;
    options.modular.max_tree_leaves = 16;
    options.modular.source_ma_search = SearchOptions {
        statistical_race: true,
        ..Default::default()
    };
    let mut policy = SamplingPolicy::from_source(&SourcePolicy {
        speed_tier: 0,
        predictor: Predictor::Variable,
        ..Default::default()
    })
    .unwrap();
    // The source race needs at least 262144 distinct records and bounded bins.
    policy.nb_repeats = 1.0;
    policy.max_property_values = 32;
    options.modular.sampling = Some(policy);
    let input = scratch.0.join("source-ma-race.jxl");
    fs::write(&input, encode(&image, &options).unwrap()).unwrap();
    for decoder in &decoders {
        let output = scratch.0.join("source-ma-race.png");
        run_checked(
            Command::new(decoder)
                .arg(&input)
                .arg(&output)
                .arg("--bits_per_sample=16")
                .arg("--num_threads=1")
                .arg("--quiet"),
            &scratch.0,
            "source MA statistical race",
        );
        assert_png_samples(&output, &image, "source MA statistical race");
        fs::remove_file(output).unwrap();
    }
}

fn check_source_ma_tools(profile: u8) {
    use rs_jxl::modular::{
        predictor::Predictor,
        sample::{SamplingPolicy, SourcePolicy},
    };
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for bits in [8, 16] {
        for layout in [PixelLayout::Gray, PixelLayout::Rgba] {
            let image = make_image(&Case {
                width: 65,
                height: 49,
                layout,
                bits,
                pattern: Pattern::Mixed,
            });
            for shared in [false, true] {
                let mut options = EncodeOptions {
                    threads: if shared { 4 } else { 1 },
                    shared_tree: shared,
                    ..Default::default()
                };
                options.modular.predictor = Predictor::Variable;
                options.modular.source_predictor_dedup = true;
                options.modular.source_property_dedup = true;
                if profile != 0 {
                    use rs_jxl::modular::ma_train::{
                        DenseHistogram, HistogramOptions, PropertyBins,
                    };
                    options.modular.source_ma_histograms = HistogramOptions {
                        shards: true,
                        padded: true,
                        dense: DenseHistogram::Extended,
                        property_bins: PropertyBins::All,
                        sparse: true,
                        paired: true,
                    };
                }
                use rs_jxl::modular::ma_train::{
                    ClippedEntropyMode, DescriptionCost, ScoringOptions,
                };
                options.modular.source_ma_scoring = match profile {
                    2 => ScoringOptions {
                        count_log: true,
                        description: DescriptionCost::Half,
                        ..Default::default()
                    },
                    3 => ScoringOptions {
                        clipped: ClippedEntropyMode::Sliding,
                        description: DescriptionCost::Full,
                        ..Default::default()
                    },
                    4 => ScoringOptions {
                        fast_entropy: true,
                        description: DescriptionCost::Quarter,
                        ..Default::default()
                    },
                    _ => ScoringOptions::default(),
                };
                if profile >= 5 {
                    use rs_jxl::modular::ma_train::{QuantileSplits, SearchOptions};
                    options.modular.source_ma_search = SearchOptions {
                        quantile_splits: if profile == 7 {
                            QuantileSplits::SixtyFour
                        } else {
                            QuantileSplits::ThirtyTwo
                        },
                        refine_quantiles: profile == 6,
                        ..Default::default()
                    };
                }
                options.modular.max_tree_leaves = rs_jxl::modular::ma_train::MAX_LEAVES;
                options.modular.sampling = Some(
                    SamplingPolicy::from_source(&SourcePolicy {
                        speed_tier: 0,
                        predictor: Predictor::Variable,
                        ..Default::default()
                    })
                    .unwrap(),
                );
                if profile >= 5 {
                    let policy = options.modular.sampling.as_mut().unwrap();
                    policy.nb_repeats = 1.0;
                    policy.max_property_values = 256;
                }
                let label =
                    format!("source MA profile={profile} {layout:?} bits={bits} shared={shared}");
                let input = scratch.0.join("source-ma-dedup.jxl");
                fs::write(
                    &input,
                    encode(&image, &options).unwrap_or_else(|e| panic!("{label}: {e}")),
                )
                .unwrap();
                for decoder in &decoders {
                    let output = scratch.0.join("source-ma-dedup.png");
                    run_checked(
                        Command::new(decoder)
                            .arg(&input)
                            .arg(&output)
                            .arg(format!("--bits_per_sample={bits}"))
                            .arg("--num_threads=1")
                            .arg("--quiet"),
                        &scratch.0,
                        &label,
                    );
                    assert_png_samples(&output, &image, &label);
                    fs::remove_file(output).unwrap();
                }
            }
        }
    }
}

fn channels(layout: PixelLayout) -> usize {
    match layout {
        PixelLayout::Gray => 1,
        PixelLayout::GrayAlpha => 2,
        PixelLayout::Rgb => 3,
        PixelLayout::Rgba => 4,
    }
}

fn make_image(case: &Case) -> Image {
    let count = channels(case.layout);
    let has_alpha = matches!(case.layout, PixelLayout::GrayAlpha | PixelLayout::Rgba);
    let maximum: u16 = if case.bits == 8 { 255 } else { 65535 };
    let mut state = 0x6d2b_79f5_u32 ^ case.width ^ case.height.rotate_left(13);
    let mut samples = Vec::with_capacity(case.width as usize * case.height as usize * count);
    for y in 0..case.height {
        for x in 0..case.width {
            for channel in 0..count {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let value = match case.pattern {
                    Pattern::Zero => 0,
                    Pattern::Maximum => maximum,
                    Pattern::HiddenMaximum => {
                        if has_alpha && channel == count - 1 {
                            0
                        } else {
                            maximum
                        }
                    }
                    Pattern::Noise => {
                        if y == 0 && x < 35 && channel < 3 {
                            // First-row Co pulses cover tokens 0..18, including
                            // -1 and the full red-to-blue 16-bit chroma swing.
                            let rgb = match x {
                                1..=31 if x % 2 == 1 => [1_u16 << ((x - 1) / 2), 0, 0],
                                33 => [maximum, 0, 0],
                                34 => [0, 0, maximum],
                                _ => [0; 3],
                            };
                            rgb[channel]
                        } else {
                            (state as u16) & maximum
                        }
                    }
                    Pattern::Mixed if has_alpha && channel == count - 1 => {
                        match (y * case.width + x) % 4 {
                            0 => 0,
                            1 => maximum,
                            2 => 1,
                            _ => maximum / 2,
                        }
                    }
                    Pattern::Mixed => match (x + y) % 8 {
                        // Alpha-bearing cases start with transparent, nonzero,
                        // unequal colors. Never mask hidden samples in comparisons.
                        0 => [1, maximum, maximum / 2 + 1][channel],
                        1 => 0,
                        2 => maximum,
                        3 if case.bits == 16 => 256 + channel as u16,
                        3 => 128 + channel as u16,
                        4 => ((x * 257 + y * 131 + channel as u32 * 73) as u16) & maximum,
                        _ => (state as u16) & maximum,
                    },
                };
                samples.push(value);
            }
        }
    }
    let pixels = if case.bits == 8 {
        PixelData::U8(samples.into_iter().map(|value| value as u8).collect())
    } else {
        PixelData::U16(samples)
    };
    let image =
        Image::new(case.width, case.height, case.layout, pixels).expect("valid generated image");
    if matches!(case.pattern, Pattern::Noise) {
        assert_noise_prefix_covers_all_tokens(&image);
    }
    image
}

fn assert_noise_prefix_covers_all_tokens(image: &Image) {
    let PixelData::U16(samples) = image.pixels() else {
        panic!("full-range noise fixture requires 16-bit samples");
    };
    assert_eq!(image.layout(), PixelLayout::Rgba);
    assert!(image.width() >= 35);
    // In the first group row, clamped gradient predicts the left value. Co is
    // R-B, so these deltas audit fixture coverage without parsing JXL output.
    let mut previous = 0_i32;
    let mut seen = [false; 19];
    for pixel in samples.chunks_exact(4).take(35) {
        let co = i32::from(pixel[0]) - i32::from(pixel[2]);
        let delta = co - previous;
        let packed = if delta < 0 {
            delta.unsigned_abs() * 2 - 1
        } else {
            delta as u32 * 2
        };
        // HybridUint(0, 0, 0) uses the unsigned residual's bit length as token.
        seen[(u32::BITS - packed.leading_zeros()) as usize] = true;
        previous = co;
    }
    assert!(
        seen.iter().all(|&present| present),
        "noise fixture misses tokens: {seen:?}"
    );
}

fn assert_png_samples(path: &Path, image: &Image, context: &str) {
    let file = File::open(path).unwrap_or_else(|error| panic!("{context}: PNG missing: {error}"));
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder
        .read_info()
        .unwrap_or_else(|error| panic!("{context}: PNG header: {error}"));
    let color = match image.layout() {
        PixelLayout::Gray => png::ColorType::Grayscale,
        PixelLayout::GrayAlpha => png::ColorType::GrayscaleAlpha,
        PixelLayout::Rgb => png::ColorType::Rgb,
        PixelLayout::Rgba => png::ColorType::Rgba,
    };
    let depth = match image.bits() {
        8 => png::BitDepth::Eight,
        16 => png::BitDepth::Sixteen,
        bits => panic!("unexpected generated bit depth {bits}"),
    };
    let info = reader.info();
    assert_eq!(
        (info.width, info.height),
        (image.width(), image.height()),
        "{context}"
    );
    assert_eq!(info.color_type, color, "{context}: PNG channels");
    assert_eq!(info.bit_depth, depth, "{context}: PNG sample depth");
    assert!(
        info.animation_control.is_none(),
        "{context}: unexpected animation"
    );

    let expected: Vec<u16> = match image.pixels() {
        PixelData::U8(samples) => samples.iter().map(|&value| u16::from(value)).collect(),
        PixelData::U16(samples) => samples.clone(),
        _ => panic!("PNG oracle only accepts U8/U16 fixtures"),
    };
    let byte_count = expected.len() * usize::from(image.bits() / 8);
    assert_eq!(
        reader.output_buffer_size(),
        Some(byte_count),
        "{context}: PNG size"
    );
    let mut bytes = vec![0; byte_count];
    let output = reader
        .next_frame(&mut bytes)
        .unwrap_or_else(|error| panic!("{context}: PNG pixels: {error}"));
    assert_eq!(
        output.buffer_size(),
        byte_count,
        "{context}: decoded byte count"
    );
    assert_eq!(output.color_type, color, "{context}: decoded channels");
    assert_eq!(output.bit_depth, depth, "{context}: decoded depth");
    let actual: Vec<u16> = if depth == png::BitDepth::Sixteen {
        // PNG samples are network byte order, not the host's native endian order.
        bytes
            .chunks_exact(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect()
    } else {
        bytes.into_iter().map(u16::from).collect()
    };
    assert_eq!(actual.len(), expected.len(), "{context}: sample count");
    if let Some((index, (actual, expected))) = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        let count = channels(image.layout());
        let pixel = index / count;
        panic!(
            "{context}: sample mismatch at ({}, {}), channel {}: expected {expected}, got {actual}",
            pixel % image.width() as usize,
            pixel / image.width() as usize,
            index % count
        );
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn lossless_pixels_match_official_djxl() {
    // Missing configuration is a failure when explicitly invoked, not a skip.
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for decoder in &decoders {
        let context = format!("{} --version", decoder.display());
        let banner = run_checked(Command::new(decoder).arg("--version"), &scratch.0, &context);
        assert!(
            banner.contains("djxl v"),
            "{context}: not a djxl version banner: {banner}"
        );
        eprintln!("{context}\n{banner}");
    }

    for (case_index, case) in cases().iter().enumerate() {
        let image = make_image(case);
        for threads in [1, 2] {
            let context = format!(
                "{}x{} {:?} {}-bit {:?}, encoder threads={threads}",
                case.width, case.height, case.layout, case.bits, case.pattern
            );
            let options = EncodeOptions {
                threads,
                max_output_bytes: 8 * 1024 * 1024,
                ..Default::default()
            };
            let encoded = encode(&image, &options)
                .unwrap_or_else(|error| panic!("{context}: encode: {error:?}"));
            assert!(!encoded.is_empty(), "{context}: empty JXL stream");
            assert!(
                encoded.len() <= options.max_output_bytes,
                "{context}: output budget exceeded"
            );
            let input = scratch.0.join(format!("case-{case_index}-t{threads}.jxl"));
            fs::write(&input, encoded).unwrap();
            for (decoder_index, decoder) in decoders.iter().enumerate() {
                // Unique names prevent a failed decoder from reusing an old PNG.
                let output = scratch
                    .0
                    .join(format!("case-{case_index}-t{threads}-d{decoder_index}.png"));
                let context = format!("{context}, decoder={}", decoder.display());
                let mut command = Command::new(decoder);
                command
                    .arg(&input)
                    .arg(&output)
                    .arg(format!("--bits_per_sample={}", image.bits()))
                    .arg("--num_threads=1")
                    .arg("--quiet");
                run_checked(&mut command, &scratch.0, &context);
                assert_png_samples(&output, &image, &context);
                fs::remove_file(output).unwrap();
            }
            // Each thread count is checked against the original samples; no
            // equality constraint is imposed on the compressed bytes or hashes.
            fs::remove_file(input).unwrap();
        }
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn modular_tools_match_official_djxl() {
    use rs_jxl::entropy::stream::Coding;
    use rs_jxl::modular::predictor::Predictor;
    use rs_jxl::tools::lz77::Lz77Mode;
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let mut variants = Vec::new();
    for kind in 0..42 {
        let mut o = EncodeOptions::default();
        o.modular.rct = Some(kind);
        variants.push((format!("rct-{kind}"), o));
    }
    for shift in 0..4 {
        let o = EncodeOptions {
            group_size_shift: shift,
            ..Default::default()
        };
        variants.push((format!("group-shift-{shift}"), o));
    }
    for palette in [None, Some(1024)] {
        let mut o = EncodeOptions {
            shared_tree: true,
            ..Default::default()
        };
        o.modular.palette_colors = palette;
        o.modular.previous_channels = 3;
        variants.push((format!("shared-tree-{palette:?}"), o));
    }
    for delta in [None, Some(Predictor::Gradient)] {
        for shared in [false, true] {
            let mut o = EncodeOptions {
                global_palette: true,
                shared_tree: shared,
                ..Default::default()
            };
            o.modular.palette_colors = Some(65536);
            o.modular.palette_predictor = delta;
            variants.push((format!("global-palette-{delta:?}-shared-{shared}"), o));
        }
    }
    for candidates in [4, 19] {
        let mut o = EncodeOptions::default();
        o.modular.rct_candidates = Some(candidates);
        o.modular.weighted_candidates = Some(5);
        variants.push((format!("auto-rct-{candidates}-wp-5"), o));
    }
    for shared in [false, true] {
        let mut o = EncodeOptions {
            shared_tree: shared,
            ..Default::default()
        };
        o.modular.max_tree_leaves = 128;
        o.modular.sampling = Some(
            rs_jxl::modular::sample::SamplingPolicy::from_source(
                &rs_jxl::modular::sample::SourcePolicy {
                    speed_tier: 3,
                    learning_fraction: 0.1,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        variants.push((format!("source-sampling-shared-{shared}"), o));
    }
    for kind in [
        rs_jxl::modular::encode::FixedTree::Zero,
        rs_jxl::modular::encode::FixedTree::Weighted,
        rs_jxl::modular::encode::FixedTree::Gradient,
    ] {
        let mut o = EncodeOptions {
            shared_tree: true,
            ..Default::default()
        };
        o.modular.fixed_tree = Some(kind);
        variants.push((format!("fixed-tree-{kind:?}"), o));
    }
    for mode in 0..=6 {
        for shared in [false, true] {
            let mut o = EncodeOptions {
                shared_tree: shared,
                ..Default::default()
            };
            o.modular.rct = None;
            o.modular.rct_candidates = Some(4);
            o.modular.palette_policy = Some(rs_jxl::modular::palette::PaletteOptions {
                mode: rs_jxl::modular::palette::PaletteMode::from_source_mode(mode).unwrap(),
                palette_colors: 10000,
                channel_colors_pre_transform_percent: 100.0,
                channel_colors_percent: 100.0,
                ..Default::default()
            });
            variants.push((format!("source-palette-{mode}-shared-{shared}"), o));
        }
    }
    for method in [
        rs_jxl::tools::lz77::SourceLz77Method::Lz77b1w3t,
        rs_jxl::tools::lz77::SourceLz77Method::Lz77b3w3t,
        rs_jxl::tools::lz77::SourceLz77Method::Optc8,
        rs_jxl::tools::lz77::SourceLz77Method::Optc256,
    ] {
        for shared in [false, true] {
            let mut o = EncodeOptions {
                shared_tree: shared,
                ..Default::default()
            };
            let mut config = rs_jxl::tools::lz77::SourceConfig::new(method);
            config.length_uint_config = rs_jxl::entropy::HybridUintConfig::new(4, 0, 0).unwrap();
            config.min_symbol = Some(224);
            config.max_match_length = (1 << 20) + 2;
            o.modular.source_lz77 = Some(config);
            variants.push((format!("source-lz-{method:?}-shared-{shared}"), o));
        }
    }
    for p in 0..14 {
        let mut o = EncodeOptions::default();
        o.modular.predictor = Predictor::try_from(p).unwrap();
        o.modular.max_tree_leaves = 1;
        variants.push((format!("predictor-{p}"), o));
    }
    for preset in 0..5 {
        let mut o = EncodeOptions::default();
        o.modular.weighted_preset = preset;
        o.modular.predictor = Predictor::Weighted;
        variants.push((format!("weighted-{preset}"), o));
    }
    for coding in [Coding::Prefix, Coding::Ans, Coding::Auto] {
        let mut o = EncodeOptions::default();
        o.modular.entropy = coding;
        variants.push((format!("entropy-{coding:?}"), o));
    }
    for mode in [
        Lz77Mode::Rle,
        Lz77Mode::Greedy,
        Lz77Mode::Lazy,
        Lz77Mode::Optimal,
    ] {
        let mut o = EncodeOptions::default();
        o.modular.lz77_mode = mode;
        variants.push((format!("lz77-{mode:?}"), o));
    }
    for palette in [None, Some(70000)] {
        for passes in [
            vec![],
            vec![true],
            vec![false],
            vec![true, false],
            vec![false, true],
            vec![true, false, true, false, true, false],
        ] {
            let mut o = EncodeOptions::default();
            o.modular.palette_colors = palette;
            o.modular.squeeze = passes.clone();
            variants.push((format!("palette-{palette:?}-squeeze-{passes:?}"), o));
        }
    }
    for predictor in [
        Predictor::Zero,
        Predictor::Left,
        Predictor::Gradient,
        Predictor::Average4,
    ] {
        let mut o = EncodeOptions::default();
        o.modular.palette_colors = Some(65536);
        o.modular.palette_predictor = Some(predictor);
        variants.push((format!("delta-palette-{predictor:?}"), o));
    }
    let mut o = EncodeOptions::default();
    o.modular.predictor = Predictor::Variable;
    o.modular.lz77 = false;
    o.modular.previous_channels = 3;
    variants.push(("all-predictors-no-lz".into(), o));
    let fixtures = [
        Case {
            width: 19,
            height: 17,
            layout: PixelLayout::Rgba,
            bits: 8,
            pattern: Pattern::Mixed,
        },
        Case {
            width: 19,
            height: 17,
            layout: PixelLayout::Rgba,
            bits: 16,
            pattern: Pattern::Mixed,
        },
        Case {
            width: 257,
            height: 9,
            layout: PixelLayout::Rgb,
            bits: 8,
            pattern: Pattern::Mixed,
        },
        Case {
            width: 9,
            height: 7,
            layout: PixelLayout::GrayAlpha,
            bits: 16,
            pattern: Pattern::Mixed,
        },
        Case {
            width: 5,
            height: 3,
            layout: PixelLayout::Rgb,
            bits: 8,
            pattern: Pattern::Zero,
        },
    ];
    let mut checks = 0;
    for (name, options) in &variants {
        for case in &fixtures {
            let image = make_image(case);
            let context = format!(
                "{name} {}x{} {:?} {}bit",
                case.width, case.height, case.layout, case.bits
            );
            let bytes = encode(&image, options).unwrap_or_else(|e| panic!("{context}: {e}"));
            let input = scratch.0.join("input.jxl");
            fs::write(&input, bytes).unwrap();
            for decoder in &decoders {
                let output = scratch.0.join("output.png");
                run_checked(
                    Command::new(decoder)
                        .arg(&input)
                        .arg(&output)
                        .arg(format!("--bits_per_sample={}", case.bits))
                        .arg("--num_threads=1")
                        .arg("--quiet"),
                    &scratch.0,
                    &context,
                );
                assert_png_samples(&output, &image, &context);
                fs::remove_file(output).unwrap();
                checks += 1;
            }
        }
    }
    eprintln!("{checks} forced-tool official decoder pixel checks passed");
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn patch_reference_frames_match_official_djxl() {
    use rs_jxl::tools::patches::{self, PatchOptions};
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for width in [192usize, 512] {
        for bits in [8, 16] {
            for layout in [
                PixelLayout::Gray,
                PixelLayout::GrayAlpha,
                PixelLayout::Rgb,
                PixelLayout::Rgba,
            ] {
                let height = 64usize;
                let count = layout.channels();
                let mut values = Vec::new();
                let mut planes = vec![Vec::new(); count];
                for y in 0..height {
                    for x in 0..width {
                        let gx = x % 24;
                        let gy = y % 24;
                        let glyph = (5..14).contains(&gx)
                            && (5..18).contains(&gy)
                            && (gx < 7 || gy < 7 || gx >= 12 || gy >= 16);
                        let color = if glyph {
                            [11u16, 27, 59, 128]
                        } else {
                            [233, 197, 163, 0]
                        };
                        for (c, plane) in planes.iter_mut().enumerate() {
                            let idx = if layout.has_alpha() && c + 1 == count {
                                3
                            } else {
                                c
                            };
                            let value = color[idx] * if bits == 16 { 257 } else { 1 };
                            values.push(value);
                            plane.push(i32::from(value));
                        }
                    }
                }
                let plan = patches::discover(
                    patches::ImageView {
                        width,
                        height,
                        bit_depth: bits,
                        color_channels: if count < 3 { 1 } else { 3 },
                        planes: &planes,
                    },
                    &PatchOptions::default(),
                )
                .unwrap();
                assert!(
                    plan.is_some(),
                    "fixture must activate patches, not silently use direct encoding"
                );
                let pixels = if bits == 8 {
                    PixelData::U8(values.into_iter().map(|v| v as u8).collect())
                } else {
                    PixelData::U16(values)
                };
                let image = Image::new(width as u32, height as u32, layout, pixels).unwrap();
                for shared_tree in [false, true] {
                    let options = EncodeOptions {
                        threads: 2,
                        shared_tree,
                        patches: Some(PatchOptions::default()),
                        ..Default::default()
                    };
                    let context = format!(
                        "patch atlas {width}x{height} {layout:?} {bits}bit shared={shared_tree}"
                    );
                    let bytes =
                        encode(&image, &options).unwrap_or_else(|e| panic!("{context}: {e}"));
                    let input = scratch.0.join("patch.jxl");
                    fs::write(&input, bytes).unwrap();
                    for decoder in &decoders {
                        let output = scratch.0.join("patch.png");
                        run_checked(
                            Command::new(decoder)
                                .arg(&input)
                                .arg(&output)
                                .arg(format!("--bits_per_sample={bits}"))
                                .arg("--num_threads=1")
                                .arg("--quiet"),
                            &scratch.0,
                            &context,
                        );
                        assert_png_samples(&output, &image, &context);
                        fs::remove_file(output).unwrap();
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn metadata_and_extra_channels_match_official_djxl() {
    use rs_jxl::color::{ColorEncoding, ColorSpace};
    use rs_jxl::metadata::{ExtraChannelInfo, ExtraChannelType, Orientation};
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    // Generated by LittleCMS via Pillow ImageCms.createProfile("sRGB"). The
    // embedded profile states "No copyright, use freely"; this is test data.
    let hex = include_str!("srgb_profile.hex").trim();
    let profile: Vec<_> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    let base = make_image(&Case {
        width: 259,
        height: 7,
        layout: PixelLayout::Rgba,
        bits: 16,
        pattern: Pattern::Mixed,
    });
    let count = base.width() as usize * base.height() as usize;
    let extras: Vec<_> = (0..3)
        .map(|c| {
            PixelData::U16(
                (0..count)
                    .map(|i| ((i * 271 + c * 997) % 65536) as u16)
                    .collect(),
            )
        })
        .collect();
    let mut metadata = base.metadata().clone();
    metadata.color_encoding = ColorEncoding::icc(ColorSpace::Rgb, profile.clone()).unwrap();
    for kind in [
        ExtraChannelType::Depth,
        ExtraChannelType::SelectionMask,
        ExtraChannelType::Optional,
    ] {
        let mut info = ExtraChannelInfo::new(kind, 16).unwrap();
        info.name = format!("test-{kind:?}");
        metadata.extra_channels.push(info);
    }
    let image = base
        .clone()
        .with_metadata(metadata, extras.clone())
        .unwrap();
    for shared_tree in [false, true] {
        let data = encode(
            &image,
            &EncodeOptions {
                shared_tree,
                threads: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let input = scratch.0.join("meta.jxl");
        fs::write(&input, data).unwrap();
        for decoder in &decoders {
            let output = scratch.0.join("meta.png");
            let icc = scratch.0.join("original.icc");
            run_checked(
                Command::new(decoder)
                    .arg(&input)
                    .arg(&output)
                    .arg("--bits_per_sample=16")
                    .arg("--output_extra_channels")
                    .arg(format!("--orig_icc_out={}", icc.display()))
                    .arg("--num_threads=1")
                    .arg("--quiet"),
                &scratch.0,
                "metadata and extras",
            );
            assert_png_samples(&output, &base, "main pixels with ICC and extras");
            assert_eq!(fs::read(&icc).unwrap(), profile, "original ICC bytes");
            for (i, extra) in extras.iter().enumerate() {
                let expected = Image::new(
                    base.width(),
                    base.height(),
                    PixelLayout::Gray,
                    extra.clone(),
                )
                .unwrap();
                let path = [
                    scratch.0.join(format!("meta-ec{}.png", i + 1)),
                    scratch.0.join(format!("meta.png-ec{}.png", i + 1)),
                ]
                .into_iter()
                .find(|p| p.is_file())
                .unwrap_or_else(|| scratch.0.join(format!("meta-ec{}.png", i + 1)));
                assert!(
                    path.is_file(),
                    "expected {} but produced {:?}",
                    path.display(),
                    fs::read_dir(&scratch.0)
                        .unwrap()
                        .map(|p| p.unwrap().file_name())
                        .collect::<Vec<_>>()
                );
                assert_png_samples(&path, &expected, "extra channel samples");
                fs::remove_file(path).unwrap();
            }
            fs::remove_file(output).unwrap();
            fs::remove_file(icc).unwrap();
        }
    }
    let PixelData::U16(original) = base.pixels() else {
        unreachable!()
    };
    for orientation in 1..=8 {
        let mut metadata = base.metadata().clone();
        metadata.orientation = Orientation::try_from(orientation).unwrap();
        let image = base.clone().with_metadata(metadata, Vec::new()).unwrap();
        let data = encode(&image, &EncodeOptions::default()).unwrap();
        let input = scratch.0.join("orientation.jxl");
        fs::write(&input, data).unwrap();
        let w = base.width() as usize;
        let h = base.height() as usize;
        let (ow, oh) = if orientation >= 5 { (h, w) } else { (w, h) };
        let mut values = vec![0; original.len()];
        for y in 0..h {
            for x in 0..w {
                let (ox, oy) = match orientation {
                    1 => (x, y),
                    2 => (w - 1 - x, y),
                    3 => (w - 1 - x, h - 1 - y),
                    4 => (x, h - 1 - y),
                    5 => (y, x),
                    6 => (h - 1 - y, x),
                    7 => (h - 1 - y, w - 1 - x),
                    8 => (y, w - 1 - x),
                    _ => unreachable!(),
                };
                values[(oy * ow + ox) * 4..(oy * ow + ox + 1) * 4]
                    .copy_from_slice(&original[(y * w + x) * 4..(y * w + x + 1) * 4]);
            }
        }
        let expected =
            Image::new(ow as u32, oh as u32, base.layout(), PixelData::U16(values)).unwrap();
        for decoder in &decoders {
            let output = scratch.0.join("orientation.png");
            run_checked(
                Command::new(decoder)
                    .arg(&input)
                    .arg(&output)
                    .arg("--bits_per_sample=16")
                    .arg("--num_threads=1")
                    .arg("--quiet"),
                &scratch.0,
                &format!("orientation {orientation}"),
            );
            assert_png_samples(&output, &expected, "oriented samples");
            fs::remove_file(output).unwrap();
        }
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL path list of official djxl decoders"]
fn native_fast_lossless_matches_official_djxl() {
    use rs_jxl::scheduler::native::encode_native_candidate;
    use rs_jxl::scheduler::params::{CandidateParams, Override};
    use rs_jxl::scheduler::{NativeBackendOptions, NativeImplementation};

    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for bits in [8, 16] {
        for layout in [
            PixelLayout::Gray,
            PixelLayout::GrayAlpha,
            PixelLayout::Rgb,
            PixelLayout::Rgba,
        ] {
            let image = make_image(&Case {
                width: 263,
                height: 13,
                layout,
                bits,
                pattern: Pattern::Mixed,
            });
            for threads in [1, 4] {
                let params = CandidateParams {
                    speed_tier: 9,
                    patches: Override::Off,
                    ..Default::default()
                };
                let options = EncodeOptions {
                    threads,
                    ..Default::default()
                };
                let encoded = encode_native_candidate(
                    &params,
                    &image,
                    &options,
                    NativeBackendOptions::default(),
                )
                .unwrap();
                assert_eq!(
                    encoded.artifact.implementation,
                    NativeImplementation::FastLossless
                );
                let input = scratch.0.join("fast.jxl");
                fs::write(&input, encoded.bytes).unwrap();
                let label = format!("FJXL/{bits}/{layout:?}/threads={threads}");
                for decoder in &decoders {
                    let output = scratch.0.join("fast.png");
                    run_checked(
                        Command::new(decoder)
                            .arg(&input)
                            .arg(&output)
                            .arg(format!("--bits_per_sample={bits}"))
                            .arg("--num_threads=1")
                            .arg("--quiet"),
                        &scratch.0,
                        &label,
                    );
                    assert_png_samples(&output, &image, &label);
                    fs::remove_file(output).unwrap();
                }
            }
        }
    }
}

#[test]
#[ignore = "requires official djxl; Level 10 thin images beyond the Level 5 axis bound"]
fn level10_long_axis_matches_official_djxl() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for (width, height) in [(262145, 1), (1, 262145)] {
        let image = make_image(&Case {
            width,
            height,
            layout: PixelLayout::Gray,
            bits: 8,
            pattern: Pattern::Mixed,
        });
        let options = EncodeOptions {
            threads: 4,
            group_size_shift: 3,
            ..Default::default()
        };
        let data = encode(&image, &options).unwrap();
        assert!(data.windows(5).any(|bytes| bytes == b"jxll\x0a"));
        let input = scratch.0.join("level10.jxl");
        fs::write(&input, data).unwrap();
        for decoder in &decoders {
            let output = scratch.0.join("level10.png");
            run_checked(
                Command::new(decoder)
                    .arg(&input)
                    .arg(&output)
                    .arg("--bits_per_sample=8")
                    .arg("--num_threads=1")
                    .arg("--quiet"),
                &scratch.0,
                "Level 10 long axis",
            );
            assert_png_samples(&output, &image, "Level 10 long axis");
            fs::remove_file(output).unwrap();
        }
        let mut sink = Vec::new();
        let limited = EncodeOptions {
            max_pixels: 262144,
            ..options
        };
        assert!(rs_jxl::encode_to(&image, &limited, &mut sink).is_err());
        assert!(sink.is_empty());
    }
}
