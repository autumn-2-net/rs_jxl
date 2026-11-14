use super::*;
use crate::modular::{
    predictor::Predictor,
    sample::{GatherMode, GatherOptions, SamplingPolicy, SourcePolicy},
    source_options::{SourceBuffering, SourcePatchReferencePolicy},
};
use crate::{PixelData, PixelLayout};

fn source_options() -> EncodeOptions {
    let sampling = SourcePolicy {
        predictor: Predictor::Gradient,
        ..Default::default()
    };
    let template = ModularOptions {
        predictor: Predictor::Gradient,
        rct: None,
        max_tree_leaves: 64,
        lz77: false,
        ..Default::default()
    };
    EncodeOptions {
        threads: 4,
        source_patches: Some(Default::default()),
        source_patch_reference: Some(SourcePatchReferencePolicy {
            modular: template.clone(),
            group_size_shift: None,
            buffering: SourceBuffering::FullFrame,
            sampling: sampling.clone(),
        }),
        modular: ModularOptions {
            sampling: Some(SamplingPolicy::from_source(&sampling).unwrap()),
            ..template
        },
        ..Default::default()
    }
}

fn active_image(bits: u8) -> Image {
    let (width, height) = (129, 83);
    let maximum = ((1u32 << bits) - 1) as u16;
    let mut samples = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let glyph = [(17, 17), (81, 17), (33, 53), (97, 53)]
                .iter()
                .any(|&(x0, y0)| (x0..x0 + 6).contains(&x) && (y0..y0 + 5).contains(&y));
            let value = if glyph { maximum / 4 } else { maximum };
            samples.extend([value, value, value, 0]);
        }
    }
    Image::new(
        width,
        height,
        PixelLayout::Rgba,
        if bits == 8 {
            PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
        } else {
            PixelData::U16(samples)
        },
    )
    .unwrap()
}

fn assert_default_gather(modular: &ModularOptions, workers: usize) {
    let gather = modular.source_gather.as_ref().expect("default gather");
    assert_eq!(gather.mode, GatherMode::Channels);
    assert_eq!(gather.workers, workers);
    assert_eq!(
        gather.batch_budget_bytes,
        GatherOptions::default().batch_budget_bytes
    );
    assert!(gather.sparse);
    assert_eq!(gather.stripes, 1);
    assert!(!gather.cache_hashes);
    assert!(!gather.concat_batches);
    gather.validate().unwrap();
}

fn assert_explicit_gather(modular: &ModularOptions, expected: &GatherOptions) {
    let actual = modular.source_gather.as_ref().expect("explicit gather");
    assert_eq!(actual.mode, expected.mode);
    assert_eq!(actual.workers, expected.workers);
    assert_eq!(actual.batch_budget_bytes, expected.batch_budget_bytes);
    assert_eq!(actual.stripes, expected.stripes);
    assert_eq!(actual.sparse, expected.sparse);
    assert_eq!(actual.cache_hashes, expected.cache_hashes);
    assert_eq!(actual.concat_batches, expected.concat_batches);
}

#[test]
fn actual_source_plan_defaults_both_frames_after_reference_resolution_and_caps_workers() {
    assert!(EncodeOptions::default().source_patch_parallel_gather);
    assert!(!EncodeOptions::default().has_patches());
    let image = active_image(8);
    let mut options = source_options();
    let (reference, _, dictionary) = prepare_patch_plan(&image, &options)
        .unwrap()
        .expect("fixture must actually activate Source Add");
    assert!(!dictionary.positions.is_empty());
    for threads in [2, 4, 64, 128, 256] {
        options.threads = threads;
        let (atlas, residual) = patch_frame_options(&options, &reference).unwrap();
        assert_default_gather(&atlas.modular, threads.min(64));
        assert_default_gather(&residual.modular, threads.min(64));
        assert_eq!(atlas.modular.predictor, Predictor::Gradient);
        assert!(atlas.shared_tree && atlas.global_palette);
        assert!(!atlas.has_patches());
        assert!(atlas.source_patch_reference.is_none());
        assert!(options.modular.source_gather.is_none());
        let policy = options.source_patch_reference.as_ref().unwrap();
        assert!(policy.modular.source_gather.is_none());
        assert!(policy.modular.sampling.is_none());
    }
}

#[test]
fn gating_helper_leaves_single_thread_optout_and_non_source_patch_modes_unchanged() {
    let reference = active_image(8);
    for case in 0..5 {
        let mut options = source_options();
        match case {
            0 => options.threads = 1,
            1 => options.source_patch_parallel_gather = false,
            2..=4 => {
                options.source_patches = None;
                if case == 3 {
                    options.patches = Some(Default::default());
                } else if case == 4 {
                    options.tile_patches = Some(Default::default());
                }
            }
            _ => unreachable!(),
        }
        let (atlas, residual) = patch_frame_options(&options, &reference).unwrap();
        assert!(atlas.modular.source_gather.is_none(), "case={case}");
        assert!(residual.modular.source_gather.is_none(), "case={case}");
    }
}

#[test]
fn gating_helper_requires_existing_sampling_independently_on_each_frame() {
    let reference = active_image(8);
    let mut options = source_options();
    options.modular.sampling = None;
    let (atlas, residual) = patch_frame_options(&options, &reference).unwrap();
    assert_default_gather(&atlas.modular, 4);
    assert!(residual.modular.sampling.is_none());
    assert!(residual.modular.source_gather.is_none());

    options.source_patch_reference = None;
    let (atlas, residual) = patch_frame_options(&options, &reference).unwrap();
    assert!(atlas.modular.sampling.is_none());
    assert!(atlas.modular.source_gather.is_none());
    assert!(residual.modular.source_gather.is_none());

    options.modular.sampling = source_options().modular.sampling;
    let (atlas, residual) = patch_frame_options(&options, &reference).unwrap();
    assert_default_gather(&atlas.modular, 4);
    assert_default_gather(&residual.modular, 4);
}

#[test]
fn explicit_main_and_reference_gather_settings_win_independently_even_when_opted_out() {
    let reference = active_image(8);
    let main = GatherOptions {
        mode: GatherMode::Waves,
        workers: 3,
        batch_budget_bytes: 4096,
        stripes: 2,
        sparse: false,
        cache_hashes: true,
        concat_batches: true,
    };
    let atlas_gather = GatherOptions {
        workers: 2,
        batch_budget_bytes: 8192,
        stripes: 3,
        ..main.clone()
    };
    for enabled in [false, true] {
        for threads in [1, 4] {
            for explicit_main in [false, true] {
                for explicit_reference in [false, true] {
                    let mut options = source_options();
                    options.threads = threads;
                    options.source_patch_parallel_gather = enabled;
                    options.modular.source_gather = explicit_main.then(|| main.clone());
                    options
                        .source_patch_reference
                        .as_mut()
                        .unwrap()
                        .modular
                        .source_gather = explicit_reference.then(|| atlas_gather.clone());
                    let (atlas, residual) = patch_frame_options(&options, &reference).unwrap();
                    for (modular, explicit, expected) in [
                        (&residual.modular, explicit_main, &main),
                        (&atlas.modular, explicit_reference, &atlas_gather),
                    ] {
                        if explicit {
                            assert_explicit_gather(modular, expected);
                        } else if enabled && threads > 1 {
                            assert_default_gather(modular, threads);
                        } else {
                            assert!(modular.source_gather.is_none());
                        }
                    }
                }
            }
        }
    }
    let mut inherited = source_options();
    inherited.source_patch_reference = None;
    inherited.modular.source_gather = Some(main.clone());
    let (atlas, residual) = patch_frame_options(&inherited, &reference).unwrap();
    assert_explicit_gather(&atlas.modular, &main);
    assert_explicit_gather(&residual.modular, &main);
}

#[test]
fn no_patch_result_bypasses_reference_resolution_and_default_gather_in_fullframe_routes() {
    let image = Image::new(3, 1, PixelLayout::Gray, PixelData::U8(vec![3, 9, 27])).unwrap();
    let mut options = source_options();
    options.threads = 2;
    options.modular.sampling = None;
    // Deliberately invalid for an atlas: no-plan fallback must never resolve it.
    options
        .source_patch_reference
        .as_mut()
        .unwrap()
        .modular
        .predictor = Predictor::Best;
    assert!(prepare_patch_plan(&image, &options).unwrap().is_none());
    assert!(options.modular.source_gather.is_none());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for cached in [false, true] {
        let bytes = if cached {
            let mut cache = LogicalFrameCache::default();
            let frame = crate::frame::Frame::new(&image);
            let logical = execution::with_pool(&pool, || {
                encode_logical_frame_cached_trial(
                    &image, &frame, &options, true, None, &mut cache, 0,
                )
                .unwrap()
                .into_complete()
                .unwrap()
            });
            assert!(cache.contains(&0));
            let mut stream = codestream::image_header(&image, options.max_output_bytes).unwrap();
            stream.extend(logical);
            wrap_stream(&image, stream, options.max_output_bytes).unwrap()
        } else {
            encode_with_pool(&image, &options, &pool).unwrap()
        };
        let decoded = crate::decode::decode(&bytes, &Default::default()).unwrap();
        assert_eq!(&decoded.frames[0].pixels, image.pixels());
    }
    assert!(options.modular.source_gather.is_none());
}

#[test]
fn source_patch_default_gather_and_ablation_roundtrip_native_samples_using_supplied_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for bits in [8, 16] {
        let image = active_image(bits);
        let mut options = source_options();
        options.threads = 2;
        assert!(prepare_patch_plan(&image, &options).unwrap().is_some());
        for (threads, enabled, supplied) in [
            (2, true, Some(&pool)),
            (2, false, Some(&pool)),
            (1, true, None),
            (2, true, None),
        ] {
            options.threads = threads;
            options.source_patch_parallel_gather = enabled;
            let bytes = encode_frames_in_pool(
                &image,
                &[crate::frame::Frame::new(&image)],
                &options,
                supplied,
            )
            .unwrap();
            let decoded = crate::decode::decode(&bytes, &Default::default()).unwrap();
            assert!(decoded.complete);
            assert_eq!(decoded.frames.len(), 1);
            // Include invisible RGB; compressed-byte identity is not a goal.
            assert_eq!(&decoded.frames[0].pixels, image.pixels());
            assert!(options.modular.source_gather.is_none());
        }
    }
}
