use super::*;
use crate::encode::{self, EncodeOptions};
use crate::modular::{
    encode::{FixedTree, ModularOptions},
    layout::Passes,
    orchestration,
    palette::PaletteOptions,
    predictor::Predictor,
};
use crate::{Image, PixelData, PixelLayout};
use std::sync::Barrier;

fn gray(width: u32, height: u32) -> Image {
    Image::new(
        width,
        height,
        PixelLayout::Gray,
        PixelData::U8(
            (0..width * height)
                .map(|i| i.wrapping_mul(17) as u8)
                .collect(),
        ),
    )
    .unwrap()
}

fn options(shared: bool, streaming: bool) -> EncodeOptions {
    EncodeOptions {
        group_size_shift: 0,
        shared_tree: shared,
        max_output_bytes: 1 << 20,
        modular: ModularOptions {
            rct: None,
            predictor: Predictor::Gradient,
            fixed_tree: Some(FixedTree::GradientLeaf),
            lz77: false,
            palette_policy: Some(PaletteOptions {
                groupwise: streaming,
                palette_colors: 0,
                channel_colors_percent: 0.0,
                channel_colors_pre_transform_percent: 0.0,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn layout(
    image: &Image,
    options: &EncodeOptions,
    budget: &SectionBudget,
) -> TrialEncode<Vec<Vec<u8>>> {
    execution::serial(|| {
        orchestration::encode_layout_trial(
            image,
            &options.modular,
            options.group_size_shift,
            &Passes::default(),
            false,
            options.shared_tree,
            options.max_output_bytes,
            None,
            Some(budget),
        )
    })
    .unwrap()
}

#[test]
fn finalized_sections_use_greater_equal_and_never_run_after_cutoff() {
    let budget = SectionBudget::new(Some(10), 100).unwrap();
    assert!(matches!(
        section(Some(&budget), || Ok(vec![0; 9])).unwrap(),
        TrialEncode::Complete(_)
    ));
    assert!(matches!(
        section(Some(&budget), || Ok(Vec::new())).unwrap(),
        TrialEncode::Complete(_)
    ));
    assert_eq!(budget.reached(), None);
    assert!(matches!(
        section(Some(&budget), || Ok(vec![0])).unwrap(),
        TrialEncode::Bounded { lower_bound: 10 }
    ));
    assert!(matches!(
        section(Some(&budget), || panic!("cancelled section started")).unwrap(),
        TrialEncode::Bounded { lower_bound: 10 }
    ));
    assert_eq!(*budget.bytes.lock().unwrap(), 10);
    assert_eq!(budget.sections_started.load(Ordering::Relaxed), 3);
}

#[test]
fn hard_max_and_arithmetic_errors_are_not_trial_bounds() {
    for beat in [None, Some(0), Some(100), Some(101)] {
        assert!(SectionBudget::new(beat, 100).is_none());
    }
    let budget = SectionBudget::new(Some(8), 10).unwrap();
    assert!(matches!(
        section(Some(&budget), || Ok(vec![0; 11])),
        Err(Error::Limit("frame section bytes"))
    ));
    assert_eq!(budget.reached(), None);
    let budget = SectionBudget::new(Some(usize::MAX - 1), usize::MAX).unwrap();
    budget.credit(usize::MAX).unwrap();
    assert!(matches!(
        budget.credit(1),
        Err(Error::Limit("frame section bytes"))
    ));
}

#[test]
fn allocation_writer_and_other_resource_errors_stay_errors() {
    for error in [
        Error::Limit("sample limit"),
        Error::Limit("pixel count"),
        Error::from(crate::bit_writer::BitWriterError::Allocation),
        Error::from(crate::bit_writer::BitWriterError::SizeLimit),
        Error::InvalidInput("bad trial".into()),
        Error::Encoding("injected codec error".into()),
    ] {
        let budget = SectionBudget::new(Some(1), 100).unwrap();
        let expected = error.to_string();
        assert_eq!(
            section(Some(&budget), || Err(error))
                .unwrap_err()
                .to_string(),
            expected
        );
        assert_eq!(budget.reached(), None);
        assert_eq!(*budget.bytes.lock().unwrap(), 0);
    }
}

#[test]
fn in_flight_error_or_hard_overrun_outweighs_parallel_cancellation() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for fail in [false, true] {
        let budget = SectionBudget::new(Some(8), 10).unwrap();
        let started = Barrier::new(2);
        let result = execution::with_pool(&pool, || {
            sections(Some(&budget), &[0, 1], |_, &task| {
                started.wait();
                if task == 0 {
                    return Ok(vec![0; 8]);
                }
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while budget.reached().is_none() {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "first section did not finish"
                    );
                    std::thread::yield_now();
                }
                if fail {
                    Err(Error::Limit("in-flight allocation"))
                } else {
                    Ok(vec![0; 3])
                }
            })
        });
        let expected = if fail {
            "in-flight allocation"
        } else {
            "frame section bytes"
        };
        assert!(matches!(result, Err(Error::Limit(message)) if message == expected));
        assert_eq!(budget.sections_started.load(Ordering::Relaxed), 2);
    }
}

#[test]
fn shared_global_bound_skips_ac_serialization_and_counts_each_section_once() {
    let image = gray(257, 3);
    let options = options(true, false);
    let full_budget = SectionBudget::new(Some(1 << 19), options.max_output_bytes).unwrap();
    let full = layout(&image, &options, &full_budget)
        .into_complete()
        .unwrap();
    let total: usize = full.iter().map(Vec::len).sum();
    assert_eq!(*full_budget.bytes.lock().unwrap(), total);
    assert!(full_budget.sections_started.load(Ordering::Relaxed) > 1);
    assert!(!full[0].is_empty());
    let bounded = SectionBudget::new(Some(full[0].len()), options.max_output_bytes).unwrap();
    assert!(
        matches!(layout(&image, &options, &bounded), TrialEncode::Bounded { lower_bound } if lower_bound == full[0].len())
    );
    assert_eq!(bounded.sections_started.load(Ordering::Relaxed), 1);
    let just_above = SectionBudget::new(Some(total + 1), options.max_output_bytes).unwrap();
    assert_eq!(
        layout(&image, &options, &just_above)
            .into_complete()
            .unwrap(),
        full
    );
}

#[test]
fn independent_stream_bound_skips_later_local_encodes() {
    let image = gray(257, 3);
    let options = options(false, false);
    let full_budget = SectionBudget::new(Some(1 << 19), options.max_output_bytes).unwrap();
    let full = layout(&image, &options, &full_budget)
        .into_complete()
        .unwrap();
    assert_eq!(
        *full_budget.bytes.lock().unwrap(),
        full.iter().map(Vec::len).sum::<usize>()
    );
    let beat = full[0].len() + 1;
    let bounded = SectionBudget::new(Some(beat), options.max_output_bytes).unwrap();
    assert!(
        matches!(layout(&image, &options, &bounded), TrialEncode::Bounded { lower_bound } if lower_bound == beat)
    );
    let attempts = bounded.sections_started.load(Ordering::Relaxed);
    assert!(attempts > 1);
    assert!(attempts < full_budget.sections_started.load(Ordering::Relaxed));
}

#[test]
fn streaming_bound_skips_next_area_preparation_and_remaining_local_encodes() {
    let image = gray(1025, 3);
    let options = options(true, true);
    let full_budget = SectionBudget::new(Some(1 << 19), options.max_output_bytes).unwrap();
    let full = layout(&image, &options, &full_budget)
        .into_complete()
        .unwrap();
    assert_eq!(full_budget.areas_started.load(Ordering::Relaxed), 2);
    assert_eq!(
        *full_budget.bytes.lock().unwrap(),
        full.iter().map(Vec::len).sum::<usize>()
    );
    let beat = full[0].len() + 1;
    let bounded = SectionBudget::new(Some(beat), options.max_output_bytes).unwrap();
    assert!(
        matches!(layout(&image, &options, &bounded), TrialEncode::Bounded { lower_bound } if lower_bound == beat)
    );
    assert_eq!(bounded.areas_started.load(Ordering::Relaxed), 1);
    assert!(
        bounded.sections_started.load(Ordering::Relaxed)
            < full_budget.sections_started.load(Ordering::Relaxed)
    );
}

#[test]
fn collapsed_frames_and_legacy_nonlayout_route_remain_complete() {
    for shared in [false, true] {
        let image = gray(17, 3);
        let options = options(shared, false);
        let budget = SectionBudget::new(Some(1), options.max_output_bytes).unwrap();
        assert!(
            !layout(&image, &options, &budget)
                .into_complete()
                .unwrap()
                .is_empty()
        );
        assert_eq!(*budget.bytes.lock().unwrap(), 0);
        assert!(matches!(
            encode::encode_trial_serial(&image, &options, Some(1)).unwrap(),
            TrialEncode::Complete(_)
        ));
        let image = gray(257, 3);
        let mut legacy = options;
        legacy.modular.palette_policy = None;
        assert!(matches!(
            encode::encode_trial_serial(&image, &legacy, Some(1)).unwrap(),
            TrialEncode::Complete(_)
        ));
    }
}

#[test]
fn root_trial_entrypoints_preserve_complete_outputs_and_scope_isolation() {
    let image = gray(257, 3);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for shared in [false, true] {
        let options = options(shared, false);
        let exact = encode::encode_serial(&image, &options).unwrap();
        for beat in [None, Some(exact.len() + 1), Some(options.max_output_bytes)] {
            assert_eq!(
                encode::encode_trial_serial(&image, &options, beat)
                    .unwrap()
                    .into_complete()
                    .unwrap(),
                exact
            );
        }
        for parallel in [false, true] {
            let outcome = if parallel {
                encode::encode_trial_with_pool(&image, &options, Some(1), &pool)
            } else {
                encode::encode_trial_serial(&image, &options, Some(1))
            }
            .unwrap();
            assert!(matches!(outcome, TrialEncode::Bounded { lower_bound: 1 }));
            assert!(!execution::is_serial());
            assert_eq!(encode::encode_serial(&image, &options).unwrap(), exact);
        }
    }
}

#[test]
fn trial_entrypoints_keep_early_validation_and_hard_max_precedence() {
    let image = gray(257, 3);
    let mut options = options(true, false);
    options.max_pixels = 1;
    assert!(matches!(
        encode::encode_trial_serial(&image, &options, Some(1)),
        Err(Error::Limit("pixel count"))
    ));
    options.max_pixels = 64_000_000;
    options.threads = 0;
    assert!(matches!(
        encode::encode_trial_serial(&image, &options, Some(1)),
        Err(Error::InvalidInput(_))
    ));
    options.threads = 1;
    options.max_output_bytes = 1;
    for beat in [None, Some(0), Some(1), Some(2)] {
        assert!(matches!(
            encode::encode_trial_serial(&image, &options, beat),
            Err(Error::Limit(_))
        ));
    }
}

#[test]
fn known_retained_headers_and_frames_also_enforce_the_hard_limit() {
    let budget = SectionBudget::new(Some(8), 10).unwrap();
    budget.retain_overhead(3).unwrap();
    assert_eq!(budget.reached(), None);
    assert!(matches!(
        section(Some(&budget), || Ok(vec![0; 8])),
        Err(Error::Limit("encoded output bytes"))
    ));
    assert_eq!(budget.reached(), None);
    let budget = SectionBudget::new(Some(8), 10).unwrap();
    budget.retain_overhead(10).unwrap();
    assert_eq!(budget.reached(), None);
    assert!(matches!(budget.retain_overhead(1), Err(Error::Limit(_))));
}

#[test]
fn actual_shared_tree_sample_limit_is_not_cancellation() {
    let image = gray(257, 3);
    let mut options = options(true, false);
    options.modular.fixed_tree = None;
    options.modular.tree_samples = 64;
    options.modular.sample_limit = 1;
    assert!(matches!(
        encode::encode_trial_serial(&image, &options, Some(1)),
        Err(Error::Limit("legacy MA selected-sample budget exceeded"))
    ));
}

#[test]
fn committed_patch_atlas_and_residual_do_not_recount_frame_assembly() {
    let (width, height) = (129, 83);
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let glyph = [(17, 17), (81, 17), (33, 53), (97, 53)]
                .iter()
                .any(|&(px, py)| (px..px + 6).contains(&x) && (py..py + 5).contains(&y));
            pixels.push(if glyph { 63 } else { 255 });
        }
    }
    let image = Image::new(width, height, PixelLayout::Gray, PixelData::U8(pixels)).unwrap();
    let config = crate::tools::patches::SourcePatchConfig::default();
    let (atlas, residual, dictionary) = crate::patch_frame::prepare_source(&image, &config)
        .unwrap()
        .expect("active source Add patch fixture");
    let mut options = options(true, false);
    options.source_patches = Some(config);
    let budget = SectionBudget::new(Some(1 << 19), options.max_output_bytes).unwrap();
    let actual = execution::serial(|| {
        encode::encode_validated_frames_trial(
            &image,
            &[crate::frame::Frame::new(&image)],
            &options,
            Some(&budget),
        )
    })
    .unwrap()
    .into_complete()
    .unwrap();
    assert_eq!(actual, encode::encode_serial(&image, &options).unwrap());
    let mut expected_sections = 0;
    for (image, patches) in [(&atlas, None), (&residual, Some(&dictionary))] {
        let budget = SectionBudget::new(Some(1 << 19), options.max_output_bytes).unwrap();
        execution::serial(|| {
            orchestration::encode_layout_trial(
                image,
                &options.modular,
                0,
                &Passes::default(),
                false,
                true,
                options.max_output_bytes,
                patches,
                Some(&budget),
            )
        })
        .unwrap()
        .into_complete()
        .unwrap();
        expected_sections += budget.section_bytes();
    }
    assert_eq!(budget.section_bytes(), expected_sections);
    assert!(expected_sections > 0 && expected_sections < actual.len());
    assert!(budget.section_bytes() + *budget.overhead.lock().unwrap() <= actual.len());
    assert!(matches!(
        encode::encode_trial_serial(&image, &options, Some(1)).unwrap(),
        TrialEncode::Bounded { lower_bound: 1 }
    ));
}

#[test]
fn actual_trial_bounds_preserve_exhaustive_winner_ties_and_work_limits() {
    use crate::scheduler::{
        params::CandidateParams,
        search::{EncodedOutput, Search, SearchLimits, TrialBudget, TrialOutput},
    };
    let image = gray(1025, 3);
    let good = options(true, false);
    let mut bad = good.clone();
    bad.modular.predictor = Predictor::Zero;
    bad.modular.fixed_tree = Some(FixedTree::Zero);
    let variants = [good.clone(), bad.clone(), good, bad];
    let exact: Vec<_> = variants
        .iter()
        .map(|options| encode::encode_serial(&image, options).unwrap())
        .collect();
    assert!(
        exact[0].len() < exact[1].len(),
        "fixture must have a losing challenger"
    );
    assert_eq!(exact[0], exact[2]);
    let pixels = u64::from(image.width()) * u64::from(image.height());
    let limits = SearchLimits {
        max_full_encodes: 4,
        max_full_pixels: 4 * pixels,
        max_output_bytes: 1 << 20,
    };
    let mut search = Search::new(limits);
    let mut bounded = 0;
    for (i, options) in variants.iter().enumerate() {
        let params = CandidateParams {
            rct: Some(i as u32),
            ..Default::default()
        };
        search
            .refine(
                &image,
                1 + i as u8,
                params,
                i == 0,
                i == 3,
                &mut |_, image, trial: TrialBudget| {
                    if i == 0 || i == 3 {
                        assert!(trial.beat_bytes.is_none());
                    }
                    Ok(
                        match encode::encode_trial_serial(image, options, trial.beat_bytes)? {
                            TrialEncode::Complete(bytes) => {
                                TrialOutput::Complete(EncodedOutput { bytes, artifact: i })
                            }
                            TrialEncode::Bounded { lower_bound } => {
                                bounded += 1;
                                assert!(lower_bound <= exact[i].len());
                                TrialOutput::Bounded { lower_bound }
                            }
                        },
                    )
                },
            )
            .unwrap();
    }
    assert!(bounded > 0);
    assert!(search.trials()[3].bytes.is_some());
    assert!(matches!(
        search.refine(
            &image,
            5,
            CandidateParams::default(),
            true,
            false,
            &mut |_, _, _| -> Result<TrialOutput<usize>> { panic!("work limit bypassed") }
        ),
        Err(Error::Limit("whole-frame search work"))
    ));
    let winner = search.finish().unwrap();
    assert_eq!(winner.winner_index, 0);
    assert_eq!(winner.artifact, 0);
    assert_eq!(winner.bytes, exact[0]);
    assert_eq!(winner.full_encodes, 4);
    assert_eq!(winner.full_pixels, 4 * pixels);
}

#[test]
fn native_source_options_reach_both_full_frame_and_streaming_collectors() {
    use crate::scheduler::{
        native::{NativeBackendOptions, encode_native_candidate},
        params::{CandidateParams, Override, SourceBuffering},
    };
    let image = gray(1025, 3);
    for buffering in [SourceBuffering::Auto, SourceBuffering::FullFrame] {
        let params = CandidateParams {
            speed_tier: 8,
            buffering,
            group_size_shift: Some(0),
            patches: Override::Off,
            ..Default::default()
        };
        let full = encode_native_candidate(
            &params,
            &image,
            &EncodeOptions::default(),
            NativeBackendOptions::default(),
        )
        .unwrap();
        assert!(full.artifact.options.modular.palette_policy.is_some());
        let budget = SectionBudget::new(Some(1), full.artifact.options.max_output_bytes).unwrap();
        let result = execution::serial(|| {
            encode::encode_validated_frames_trial(
                &image,
                &[crate::frame::Frame::new(&image)],
                &full.artifact.options,
                Some(&budget),
            )
        })
        .unwrap();
        assert!(matches!(result, TrialEncode::Bounded { lower_bound: 1 }));
        assert_eq!(budget.sections_started.load(Ordering::Relaxed), 1);
        assert_eq!(
            budget.areas_started.load(Ordering::Relaxed),
            usize::from(buffering == SourceBuffering::Auto)
        );
        assert!(!full.bytes.is_empty());
    }
}

#[test]
#[ignore = "requires explicit RS_JXL_DJXL; native bounded/unbounded size and pixel oracle"]
fn native_bounded_minimum_and_pixels_match_unbounded_official_djxl() {
    use crate::scheduler::{
        native::{NativeBackendOptions, encode_native_candidate},
        params::{CandidateParams, Override, SourceBuffering},
    };
    use std::{env, fs, io::BufReader, path::PathBuf, process::Command};

    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            // Only these two files are created; never recursively remove a
            // computed path, including during an oracle assertion failure.
            let _ = fs::remove_file(self.0.join("trial.jxl"));
            let _ = fs::remove_file(self.0.join("trial.png"));
            let _ = fs::remove_dir(&self.0);
        }
    }
    let decoders: Vec<_> = env::split_paths(
        &env::var_os("RS_JXL_DJXL").expect("explicit official decoder paths required"),
    )
    .collect();
    assert!(!decoders.is_empty());
    assert!(decoders.iter().all(|path| path.is_file()));
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = env::temp_dir().join(format!(
        "rs-jxl-native-budget-{}-{stamp}",
        std::process::id(),
    ));
    fs::create_dir(&path).unwrap();
    let scratch = Scratch(path);
    // Repeated nonperiodic rows favor Top. Left retains a broad residual
    // distribution on every row, without relying on group-size overhead.
    let mut state = 0x9e37_79b9u32;
    let row: Vec<u8> = (0..1025)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect();
    let image = Image::new(
        1025,
        17,
        PixelLayout::Gray,
        PixelData::U8(row.iter().copied().cycle().take(row.len() * 17).collect()),
    )
    .unwrap();
    let PixelData::U8(expected) = image.pixels() else {
        unreachable!()
    };
    let check_pixels = |bytes: &[u8]| {
        let input = scratch.0.join("trial.jxl");
        let output = scratch.0.join("trial.png");
        fs::write(&input, bytes).unwrap();
        for decoder in &decoders {
            assert!(!output.exists(), "stale decoder output");
            let mut command = Command::new(decoder);
            for (key, _) in env::vars_os() {
                let name = key.to_string_lossy().to_ascii_uppercase();
                if name.starts_with("JXL_") || name.starts_with("RS_JXL_") {
                    command.env_remove(key);
                }
            }
            let decoded = command
                .arg(&input)
                .arg(&output)
                .args(["--bits_per_sample=8", "--num_threads=1", "--quiet"])
                .output()
                .unwrap();
            assert!(
                decoded.status.success(),
                "{}: {}",
                decoder.display(),
                String::from_utf8_lossy(&decoded.stderr)
            );
            let mut reader = png::Decoder::new(BufReader::new(fs::File::open(&output).unwrap()))
                .read_info()
                .unwrap();
            let info = reader.info();
            assert_eq!((info.width, info.height), (image.width(), image.height()));
            assert_eq!(info.color_type, png::ColorType::Grayscale);
            assert_eq!(info.bit_depth, png::BitDepth::Eight);
            assert_eq!(reader.output_buffer_size(), Some(expected.len()));
            let mut pixels = vec![0; expected.len()];
            let info = reader.next_frame(&mut pixels).unwrap();
            assert_eq!(info.buffer_size(), expected.len());
            assert_eq!(&pixels, expected);
            drop(reader);
            fs::remove_file(&output).unwrap();
        }
    };
    for buffering in [SourceBuffering::FullFrame, SourceBuffering::Auto] {
        let mut exhaustive = Vec::new();
        for predictor in [Predictor::Top, Predictor::Left] {
            let params = CandidateParams {
                // Tiers 7..=9 select predefined trees regardless of the
                // requested predictor. Tier 6 uses the explicit singleton
                // predictor as LearnTree's zero-sample seed, with no LZ77.
                speed_tier: 6,
                buffering,
                predictor: Some(predictor),
                learning_fraction: 0.0,
                group_size_shift: Some(0),
                palette_colors: 0,
                channel_colors_percent: 0.0,
                channel_colors_pre_transform_percent: 0.0,
                patches: Override::Off,
                ..Default::default()
            };
            let output = encode_native_candidate(
                &params,
                &image,
                &EncodeOptions::default(),
                NativeBackendOptions::default(),
            )
            .unwrap();
            let modular = &output.artifact.options.modular;
            assert_eq!(modular.predictor, predictor);
            assert!(modular.fixed_tree.is_none());
            assert!(modular.custom_tree.is_none());
            assert!(!modular.lz77);
            assert!(modular.source_lz77.is_none());
            assert!(!modular.source_constant_channels);
            let sampling = modular.sampling.as_ref().expect("source sampling policy");
            assert_eq!(sampling.nb_repeats, 0.0);
            assert_eq!(sampling.predictor_override, None);
            assert_eq!(
                modular.palette_policy.as_ref().unwrap().groupwise,
                buffering == SourceBuffering::Auto,
            );
            check_pixels(&output.bytes);
            exhaustive.push(output);
        }
        assert!(
            exhaustive[1].bytes.len() > exhaustive[0].bytes.len(),
            "{buffering:?}: Left must lose to Top: sizes={:?}",
            exhaustive
                .iter()
                .map(|output| output.bytes.len())
                .collect::<Vec<_>>(),
        );
        let minimum = exhaustive
            .iter()
            .map(|output| output.bytes.len())
            .min()
            .unwrap();
        let mut winner: Option<Vec<u8>> = None;
        let mut bounded = 0;
        for full in &exhaustive {
            let beat = winner.as_ref().map(Vec::len);
            // Same private native entrypoint used by null-runner full batches.
            // Errors propagate as errors; only this typed variant is a bound.
            match encode::encode_trial_serial(&image, &full.artifact.options, beat).unwrap() {
                TrialEncode::Complete(bytes) => {
                    assert_eq!(bytes.len(), full.bytes.len());
                    check_pixels(&bytes);
                    if winner.as_ref().is_none_or(|old| bytes.len() < old.len()) {
                        winner = Some(bytes);
                    }
                }
                TrialEncode::Bounded { lower_bound } => {
                    assert!(lower_bound >= beat.expect("baseline cannot be bounded"));
                    assert!(lower_bound <= full.bytes.len());
                    bounded += 1;
                }
            }
        }
        assert!(
            bounded > 0,
            "{buffering:?}: fixture must cancel a native challenger"
        );
        let winner = winner.unwrap();
        assert_eq!(winner.len(), minimum);
        check_pixels(&winner);
    }
}
