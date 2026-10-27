use rs_jxl::{
    EncodeOptions, Image, PixelData, PixelLayout,
    codec::Encoder,
    metadata::{ExtraChannelInfo, ExtraChannelType},
    multiframe::{self, AnalysisOptions, AutoOptions, Reference, Strategy, Suitability},
};

fn rgba(width: u32, height: u32, changed: &[(usize, usize)], alpha: u8) -> Image {
    let mut pixels = vec![0; width as usize * height as usize * 4];
    for pixel in pixels.chunks_exact_mut(4) {
        pixel[3] = alpha;
    }
    for &(x, y) in changed {
        pixels[(y * width as usize + x) * 4] = 31;
    }
    Image::new(width, height, PixelLayout::Rgba, PixelData::U8(pixels)).unwrap()
}

#[test]
fn exact_statistics_and_plans_match_all_reference_modes() {
    let images = vec![
        rgba(65, 65, &[], 0),
        rgba(65, 65, &[(1, 2)], 0),
        rgba(65, 65, &[(64, 64)], 0),
        rgba(65, 65, &[(64, 64)], 0),
    ];
    for reference in [
        Reference::Full,
        Reference::Previous,
        Reference::First,
        Reference::SmallestCrop,
    ] {
        let report = multiframe::analyze(
            &images,
            &AnalysisOptions {
                reference,
                ..Default::default()
            },
        )
        .unwrap();
        let planned = multiframe::plan(&images, &[0; 4], reference).unwrap();
        assert!(report.frames[0].previous_stats.is_none());
        assert!(report.frames[0].first_stats.is_none());
        assert_eq!(
            report.frames[1]
                .previous_stats
                .as_ref()
                .unwrap()
                .changed_pixels,
            1
        );
        assert_eq!(
            report.frames[2]
                .previous_stats
                .as_ref()
                .unwrap()
                .changed_pixels,
            2
        );
        assert_eq!(
            report.frames[3]
                .previous_stats
                .as_ref()
                .unwrap()
                .changed_pixels,
            0
        );
        assert_eq!(report.frames[3].previous, None);
        for (frame, actual) in report.frames.iter().zip(planned) {
            assert_eq!(frame.plan.changed, actual.changed);
            assert_eq!(frame.plan.region.width, actual.image.width());
            assert_eq!(frame.plan.region.height, actual.image.height());
            assert_eq!(
                (frame.plan.region.x as i32, frame.plan.region.y as i32),
                actual.options.origin
            );
            assert_eq!(frame.plan.source, actual.options.blend.source);
            assert!(
                frame
                    .candidates
                    .iter()
                    .all(|c| !matches!(c.strategy, Strategy::MaskPrevious | Strategy::MaskFirst))
            );
        }
        let summary = report
            .reference_modes
            .iter()
            .find(|s| s.reference == reference)
            .unwrap();
        assert_eq!(
            summary.layer_pixels,
            report
                .frames
                .iter()
                .map(|f| f.plan.layer_pixels)
                .sum::<u64>()
        );
    }
}

#[test]
fn sparse_mask_and_regions_share_actual_encoder_planning() {
    let images = vec![
        rgba(80, 72, &[], 255),
        rgba(80, 72, &[(1, 1), (65, 65)], 255),
    ];
    let auto = AutoOptions {
        tile_size: 16,
        max_rectangles: 8,
        prune: false,
        ..Default::default()
    };
    let encoded =
        multiframe::encode_auto(&images, &[0, 0], &EncodeOptions::default(), &auto).unwrap();
    let anchor = encoded.frames[0].candidates[0].encoded_bytes as f64 / (80.0 * 72.0);
    let report = multiframe::analyze(
        &images,
        &AnalysisOptions {
            tile_size: 16,
            max_rectangles: 8,
            rectangle_bytes_per_pixel: anchor,
            ..Default::default()
        },
    )
    .unwrap();
    let stats = report.frames[1].previous_stats.as_ref().unwrap();
    assert_eq!(stats.changed_pixels, 2);
    assert_eq!(stats.bbox_pixels, 65 * 65);
    assert!(stats.mask_eligible);
    assert!(stats.bbox_fill_ratio < 0.001);
    assert_eq!(
        report.frames[1].candidates.len(),
        encoded.frames[1].candidates.len()
    );
    for (analysis, actual) in report.frames[1]
        .candidates
        .iter()
        .zip(&encoded.frames[1].candidates)
    {
        assert_eq!(analysis.strategy, actual.strategy);
        assert_eq!(analysis.tile_size, actual.tile_size);
        assert_eq!(analysis.layer_count, actual.encoded_frames);
        assert_eq!(analysis.layer_pixels, actual.trial_pixels);
    }
    let mask = report.frames[1]
        .candidates
        .iter()
        .find(|c| c.strategy == Strategy::MaskPrevious)
        .unwrap();
    assert_eq!(mask.layer_pixels, stats.bbox_pixels);
    assert!(
        report.frames[1]
            .candidates
            .iter()
            .any(|c| c.layer_count == 2 && c.layer_pixels == 2)
    );
    // Decode final composited state; compressed byte identity is not an oracle.
    let decoded = rs_jxl::decode::decode(&encoded.bytes, &Default::default()).unwrap();
    assert_eq!(&decoded.frames.last().unwrap().pixels, images[1].pixels());
}

#[test]
fn identical_frames_still_need_one_pixel_and_initial_full_frame() {
    let image = rgba(10, 10, &[], 255);
    let report =
        multiframe::analyze(&[image.clone(), image.clone(), image], &Default::default()).unwrap();
    let previous = &report.reference_modes[1];
    assert_eq!(previous.layer_pixels, 102);
    assert_eq!(previous.identical_frames, 2);
    assert_eq!(previous.average_changed_ratio, 0.0);
    assert_eq!(previous.suitability, Suitability::Strong);
    assert!((previous.layer_area_ratio - 102.0 / 300.0).abs() < 1e-12);
    assert_eq!(report.frames[2].plan.reference_index, Some(1)); // ties prefer previous
    assert_eq!(report.frames[2].plan.layer_pixels, 1);
    assert_eq!(
        report.frames[1]
            .previous_stats
            .as_ref()
            .unwrap()
            .bbox_pixels,
        0
    );
}

#[test]
fn float_words_and_wide_integers_are_compared_exactly() {
    for (a, b) in [
        (
            PixelData::F32(vec![
                0.0,
                f32::from_bits(0x7fc0_0001),
                f32::from_bits(0x7fc0_0001),
            ]),
            PixelData::F32(vec![
                -0.0,
                f32::from_bits(0x7fc0_0002),
                f32::from_bits(0x7fc0_0001),
            ]),
        ),
        (
            PixelData::U16(vec![65_535, 1, 2]),
            PixelData::U16(vec![65_534, 2, 2]),
        ),
        (
            PixelData::U32(vec![0x3fff_ffff, 1, 2]),
            PixelData::U32(vec![0x3fff_fffe, 2, 2]),
        ),
    ] {
        let images = [
            Image::new(3, 1, PixelLayout::Gray, a).unwrap(),
            Image::new(3, 1, PixelLayout::Gray, b).unwrap(),
        ];
        let report = multiframe::analyze(&images, &Default::default()).unwrap();
        assert_eq!(
            report.frames[1]
                .previous_stats
                .as_ref()
                .unwrap()
                .changed_pixels,
            2
        );
        assert!(
            !report.frames[1]
                .previous_stats
                .as_ref()
                .unwrap()
                .mask_eligible
        );
    }
}

#[test]
fn extra_samples_count_union_and_subsampling_reports_full_only() {
    for shift in [0, 1] {
        let base = rgba(7, 5, &[], 255);
        let mut metadata = base.metadata().clone();
        let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
        extra.dim_shift = shift;
        metadata.extra_channels.push(extra);
        let count = if shift == 0 { 35 } else { 12 };
        let a = base
            .clone()
            .with_metadata(metadata.clone(), vec![PixelData::U8(vec![0; count])])
            .unwrap();
        let mut samples = vec![0; count];
        samples[count - 1] = 1;
        // Last pixel overlaps the changed extra sample's clipped canvas cell.
        let b = rgba(7, 5, &[(6, 4)], 255)
            .with_metadata(metadata, vec![PixelData::U8(samples)])
            .unwrap();
        let report = multiframe::analyze(&[a, b], &Default::default()).unwrap();
        let stats = report.frames[1].previous_stats.as_ref().unwrap();
        assert_eq!(stats.changed_pixels, 1);
        assert!(!stats.mask_eligible);
        assert_eq!(report.crop_supported, shift == 0);
        if shift != 0 {
            assert_eq!(report.frames[1].candidates.len(), 1);
            assert_eq!(report.frames[1].plan.layer_pixels, 35);
            assert_eq!(
                report.reference_modes[1].suitability,
                Suitability::FullFramesRequired
            );
        }
    }
}

#[test]
fn validation_and_single_frame_analysis() {
    assert!(multiframe::analyze(&[], &Default::default()).is_err());
    let a = rgba(4, 4, &[], 255);
    let b = rgba(5, 4, &[], 255);
    assert!(multiframe::analyze(&[a.clone(), b], &Default::default()).is_err());
    let mut metadata = a.metadata().clone();
    metadata.orientation = rs_jxl::metadata::Orientation::FlipHorizontal;
    let b = a.clone().with_metadata(metadata, vec![]).unwrap();
    assert!(multiframe::analyze(&[a.clone(), b], &Default::default()).is_err());
    let inputs = [a];
    for options in [
        AnalysisOptions {
            threads: 0,
            ..Default::default()
        },
        AnalysisOptions {
            threads: 257,
            ..Default::default()
        },
        AnalysisOptions {
            tile_size: 0,
            ..Default::default()
        },
        AnalysisOptions {
            extra_tile_sizes: vec![16, 0, 64],
            allow_multi_rect: false,
            ..Default::default()
        },
        AnalysisOptions {
            max_rectangles: 0,
            ..Default::default()
        },
        AnalysisOptions {
            rectangle_bytes_per_pixel: -1.0,
            ..Default::default()
        },
        AnalysisOptions {
            rectangle_bytes_per_pixel: f64::NAN,
            ..Default::default()
        },
        AnalysisOptions {
            rectangle_bytes_per_pixel: f64::INFINITY,
            ..Default::default()
        },
    ] {
        assert!(multiframe::analyze(&inputs, &options).is_err());
    }
    let report = multiframe::analyze(&inputs, &Default::default()).unwrap();
    assert_eq!(report.frame_count, 1);
    assert_eq!(report.frames[0].candidates.len(), 1);
    assert!(
        report
            .reference_modes
            .iter()
            .all(|s| s.suitability == Suitability::SingleFrame && s.layer_area_ratio == 1.0)
    );
}

#[test]
fn worker_context_is_reusable_after_failure_and_matches_serial() {
    let images = [
        rgba(513, 513, &[], 255),
        rgba(513, 513, &[(512, 512), (0, 0)], 255),
    ];
    let serial = multiframe::analyze(&images, &Default::default()).unwrap();
    let mut context = Encoder::new(4).unwrap();
    assert!(
        context
            .analyze_multiframe(&images, &Default::default())
            .is_err()
    );
    let options = AnalysisOptions {
        threads: 4,
        ..Default::default()
    };
    assert!(context.analyze_multiframe(&[], &options).is_err());
    for _ in 0..2 {
        let report = context.analyze_multiframe(&images, &options).unwrap();
        assert_eq!(report.frames[1].previous, serial.frames[1].previous);
        assert_eq!(
            report.frames[1]
                .previous_stats
                .as_ref()
                .unwrap()
                .changed_pixels,
            2
        );
        assert_eq!(
            report.frames[1]
                .candidates
                .iter()
                .map(|c| (&c.rectangles, c.strategy))
                .collect::<Vec<_>>(),
            serial.frames[1]
                .candidates
                .iter()
                .map(|c| (&c.rectangles, c.strategy))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn disabling_proposals_does_not_disable_exact_comparison() {
    let inputs = [
        rgba(80, 72, &[], 255),
        rgba(80, 72, &[(1, 1), (65, 65)], 255),
    ];
    let report = multiframe::analyze(
        &inputs,
        &AnalysisOptions {
            allow_mask: false,
            allow_multi_rect: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(report.frames[1].candidates.len(), 2);
    assert_eq!(
        report.frames[1]
            .previous_stats
            .as_ref()
            .unwrap()
            .changed_pixels,
        2
    );
    let full = multiframe::analyze(
        &inputs,
        &AnalysisOptions {
            reference: Reference::Full,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(full.frames[1].candidates.len(), 1);
    assert_eq!(
        full.frames[1]
            .previous_stats
            .as_ref()
            .unwrap()
            .changed_pixels,
        2
    );
}

#[test]
fn history_window_measures_preceding_positions_and_excludes_static_references() {
    let images: Vec<_> = [0, 17, 29, 17, 17]
        .into_iter()
        .map(|value| Image::new(3, 2, PixelLayout::Gray, PixelData::U8(vec![value; 6])).unwrap())
        .collect();
    let baseline = multiframe::analyze(&images, &Default::default()).unwrap();
    assert_eq!(baseline.history_window, 0);
    assert!(baseline.frames.iter().all(|frame| frame.history.is_empty()));
    for window in [0, 1, 2, 3, usize::MAX] {
        let report = multiframe::analyze(
            &images,
            &AnalysisOptions {
                history_window: window,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(report.history_window, window);
        for (index, frame) in report.frames.iter().enumerate() {
            let expected: Vec<_> = (index.saturating_sub(window)..index)
                .rev()
                .filter(|&reference| reference != 0 && reference + 1 != index)
                .collect();
            assert_eq!(
                frame
                    .history
                    .iter()
                    .map(|entry| entry.reference_index)
                    .collect::<Vec<_>>(),
                expected,
            );
            // Observations cannot create new candidate references or slots.
            assert_eq!(
                frame.plan.reference_index,
                baseline.frames[index].plan.reference_index
            );
            assert_eq!(
                frame.candidates.len(),
                baseline.frames[index].candidates.len()
            );
            assert!(frame.candidates.iter().all(|candidate| {
                candidate
                    .reference_index
                    .is_none_or(|reference| reference == 0 || reference + 1 == index)
            }));
        }
        if window >= 2 {
            let entry = &report.frames[3].history[0];
            assert_eq!(entry.reference_index, 1);
            assert_eq!(entry.stats.changed_pixels, 0);
            assert!(entry.improves_static_bbox);
        }
        if window >= 3 {
            let entry = &report.frames[4].history[1];
            assert_eq!(entry.reference_index, 1);
            assert_eq!(entry.stats.bbox_pixels, 0);
            assert!(!entry.improves_static_bbox); // Previous already has zero bbox.
        }
    }
}

#[test]
fn history_reports_hidden_color_float_words_and_extra_planes_exactly() {
    let hidden_base = rgba(4, 3, &[], 0);
    let hidden_changed = rgba(4, 3, &[(2, 1)], 0);
    let cases = [
        (hidden_base, hidden_changed),
        (
            Image::new(
                3,
                1,
                PixelLayout::Gray,
                PixelData::F16(vec![0, 0x7e01, 0x7e01]),
            )
            .unwrap(),
            Image::new(
                3,
                1,
                PixelLayout::Gray,
                PixelData::F16(vec![0x8000, 0x7e02, 0x7e01]),
            )
            .unwrap(),
        ),
        (
            Image::new(
                3,
                1,
                PixelLayout::Gray,
                PixelData::F32(vec![
                    0.0,
                    f32::from_bits(0x7fc0_0001),
                    f32::from_bits(0x7fc0_0001),
                ]),
            )
            .unwrap(),
            Image::new(
                3,
                1,
                PixelLayout::Gray,
                PixelData::F32(vec![
                    -0.0,
                    f32::from_bits(0x7fc0_0002),
                    f32::from_bits(0x7fc0_0001),
                ]),
            )
            .unwrap(),
        ),
    ];
    for (case, (base, changed)) in cases.into_iter().enumerate() {
        let images = [base.clone(), base.clone(), base, changed];
        let report = multiframe::analyze(
            &images,
            &AnalysisOptions {
                history_window: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let stats = &report.frames[3].history[0].stats;
        assert_eq!(stats.changed_pixels, if case == 0 { 1 } else { 2 });
        assert_eq!(
            stats.bounds,
            report.frames[3].previous_stats.as_ref().unwrap().bounds
        );
        assert!(!stats.mask_eligible);
        assert!(!report.frames[3].history[0].improves_static_bbox);
    }

    let base = rgba(7, 5, &[], 255);
    let mut metadata = base.metadata().clone();
    metadata
        .extra_channels
        .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap());
    let a = base
        .clone()
        .with_metadata(metadata.clone(), vec![PixelData::U16(vec![0; 35])])
        .unwrap();
    let mut samples = vec![0; 35];
    samples[10] = 65535;
    let b = base
        .with_metadata(metadata, vec![PixelData::U16(samples)])
        .unwrap();
    let report = multiframe::analyze(
        &[a.clone(), a.clone(), a, b],
        &AnalysisOptions {
            history_window: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(report.frames[3].history[0].stats.changed_pixels, 1);
    assert!(!report.frames[3].history[0].stats.mask_eligible);
}

#[test]
fn subsampled_history_is_observed_but_never_recommends_crop() {
    let base = rgba(7, 5, &[], 255);
    let mut metadata = base.metadata().clone();
    let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
    extra.dim_shift = 1;
    metadata.extra_channels.push(extra);
    let a = base
        .clone()
        .with_metadata(metadata.clone(), vec![PixelData::U8(vec![0; 12])])
        .unwrap();
    let b = base
        .with_metadata(metadata, vec![PixelData::U8(vec![1; 12])])
        .unwrap();
    let report = multiframe::analyze(
        &[a.clone(), b.clone(), a, b],
        &AnalysisOptions {
            history_window: usize::MAX,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!report.crop_supported);
    let entry = &report.frames[3].history[0];
    assert_eq!(entry.reference_index, 1);
    assert_eq!(entry.stats.changed_pixels, 0);
    assert!(!entry.improves_static_bbox);
    assert_eq!(report.geometry.suitability, Suitability::FullFramesRequired);
    assert_eq!(report.geometry.layer_pixels, 4 * 35);
    assert_eq!(report.geometry.layer_count, 4);
    assert_eq!(report.geometry.crop_frames, 0);
    assert_eq!(report.geometry.multi_rect_frames, 0);
    assert!(
        report
            .frames
            .iter()
            .all(|frame| frame.candidates.len() == 1)
    );
}

#[test]
fn geometry_summary_accounts_for_sparse_regions_without_changing_bbox_semantics() {
    let images = [
        rgba(80, 72, &[], 255),
        rgba(80, 72, &[(1, 1), (65, 65)], 255),
    ];
    let report = multiframe::analyze(
        &images,
        &AnalysisOptions {
            tile_size: 16,
            ..Default::default()
        },
    )
    .unwrap();
    let previous = report
        .reference_modes
        .iter()
        .find(|s| s.reference == Reference::Previous)
        .unwrap();
    assert_eq!(previous.layer_pixels, 80 * 72 + 65 * 65);
    assert_eq!(previous.suitability, Suitability::Limited);
    let frame = &report.frames[1];
    let best = &frame.candidates[frame.geometry.best_area_candidate_index];
    assert_eq!(best.layer_pixels, 2);
    assert_eq!(best.layer_count, 2);
    assert_eq!(best.tile_size, Some(16));
    assert_eq!(report.geometry.layer_pixels, 80 * 72 + 2);
    assert_eq!(report.geometry.layer_count, 3);
    assert_eq!(report.geometry.crop_frames, 1);
    assert_eq!(report.geometry.multi_rect_frames, 1);
    assert_eq!(report.geometry.suitability, Suitability::Strong);
    assert_eq!(frame.geometry.candidate_count, frame.candidates.len());
    assert_eq!(
        frame.geometry.trial_pixels,
        frame.candidates.iter().map(|c| c.layer_pixels).sum::<u64>()
    );
    assert_eq!(
        frame.geometry.trial_layers,
        frame
            .candidates
            .iter()
            .map(|c| c.layer_count)
            .sum::<usize>()
    );
    assert_eq!(
        report.geometry.trial_pixels,
        report
            .frames
            .iter()
            .map(|f| f.geometry.trial_pixels)
            .sum::<u64>()
    );
    assert_eq!(
        report.geometry.trial_layers,
        report
            .frames
            .iter()
            .map(|f| f.geometry.trial_layers)
            .sum::<usize>()
    );
}

#[test]
fn repeated_rectangle_work_is_scoped_to_one_input_and_excludes_masks() {
    let images: Vec<_> = [0, 10, 20]
        .into_iter()
        .map(|value| {
            let mut samples = vec![0; 64];
            samples[9] = value;
            Image::new(8, 8, PixelLayout::Gray, PixelData::U8(samples)).unwrap()
        })
        .collect();
    let options = AnalysisOptions {
        allow_multi_rect: false,
        ..Default::default()
    };
    let report = multiframe::analyze(&images, &options).unwrap();
    let geometry = &report.frames[2].geometry;
    assert_eq!(geometry.best_area_candidate_index, 1); // Previous wins canonical tie.
    assert_eq!(geometry.candidate_count, 3);
    assert_eq!((geometry.trial_pixels, geometry.trial_layers), (66, 3));
    let work = &geometry.unmasked_rectangles;
    assert_eq!((work.total_pixels, work.total_layers), (66, 3));
    assert_eq!((work.unique_pixels, work.unique_rectangles), (65, 2));
    assert_eq!((work.repeated_pixels, work.repeated_layers), (1, 1));
    let total = &report.geometry.unmasked_rectangles;
    assert_eq!((total.total_pixels, total.total_layers), (195, 6));
    assert_eq!((total.unique_pixels, total.unique_rectangles), (194, 5));
    assert_eq!((total.repeated_pixels, total.repeated_layers), (1, 1));
    assert_eq!(
        (report.geometry.layer_pixels, report.geometry.layer_count),
        (66, 3)
    );

    let report = multiframe::analyze(
        &[rgba(8, 8, &[], 255), rgba(8, 8, &[(1, 1)], 255)],
        &options,
    )
    .unwrap();
    let geometry = &report.frames[1].geometry;
    assert!(
        report.frames[1]
            .candidates
            .iter()
            .any(|c| c.strategy == Strategy::MaskPrevious)
    );
    assert_eq!(geometry.best_area_candidate_index, 1); // Crop before equal-area mask.
    assert_eq!((geometry.trial_pixels, geometry.trial_layers), (66, 3));
    assert_eq!(
        (
            geometry.unmasked_rectangles.total_pixels,
            geometry.unmasked_rectangles.total_layers
        ),
        (65, 2)
    );
    assert_eq!(geometry.unmasked_rectangles.repeated_layers, 0);
    assert_eq!(geometry.unmasked_rectangles.repeated_pixels, 0);
}

#[test]
fn multiscale_geometry_is_forwarded_and_equivalent_plans_are_not_counted_twice() {
    let images = [
        rgba(64, 64, &[], 255),
        rgba(64, 64, &[(1, 1), (25, 25)], 255),
    ];
    let options = AnalysisOptions {
        tile_size: 64,
        allow_mask: false,
        adaptive_rectangles: false,
        ..Default::default()
    };
    let baseline = multiframe::analyze(&images, &options).unwrap();
    let scaled = multiframe::analyze(
        &images,
        &AnalysisOptions {
            extra_tile_sizes: vec![16, 8, 64, 16],
            ..options
        },
    )
    .unwrap();
    assert_eq!(baseline.frames[1].candidates.len(), 2);
    assert_eq!(scaled.frames[1].candidates.len(), 3);
    let frame = &scaled.frames[1];
    let best = &frame.candidates[frame.geometry.best_area_candidate_index];
    assert_eq!(best.strategy, Strategy::TilesPrevious);
    assert_eq!(best.tile_size, Some(16));
    assert_eq!(best.layer_pixels, 2);
    assert_eq!(best.layer_count, 2);
    assert_eq!(frame.geometry.trial_pixels, 64 * 64 + 25 * 25 + 2);
    assert_eq!(frame.geometry.trial_layers, 4);
    assert!(frame.candidates[..2].iter().all(|c| c.tile_size.is_none()));
}
