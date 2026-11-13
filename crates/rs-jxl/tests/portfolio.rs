use rs_jxl::{
    Image, PixelData, PixelLayout,
    decode::{DecodeOptions, decode},
    scheduler::portfolio::{self, Candidate, Options},
};

fn check(image: &Image, options: &Options) -> portfolio::Output {
    let result = portfolio::encode(image, options).unwrap();
    let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
    assert!(decoded.complete);
    assert_eq!(decoded.frames.len(), 1);
    assert_eq!(&decoded.frames[0].pixels, image.pixels());
    assert_eq!(decoded.frames[0].layout, image.layout());
    assert!(result.bytes.len() <= result.report.candidates[0].full_bytes.unwrap());
    assert!(result.report.probe_encodes <= options.max_probe_encodes);
    assert!(result.report.full_encodes <= options.max_finalists + 1);
    assert!(result.report.seconds >= result.report.scout_seconds);
    for trial in &result.report.candidates {
        if let Some(lower_bound) = trial.full_lower_bound {
            assert_eq!(trial.decision, "full-bounded");
            assert!(trial.full_bytes.is_none());
            assert!(trial.error.is_none());
            assert!(trial.full_seconds.is_some());
            assert!(lower_bound >= result.bytes.len());
        }
    }
    result
}

#[test]
fn performance_switches_keep_pixels_and_work_budgets() {
    let image = Image::new(
        80,
        72,
        PixelLayout::Rgba,
        PixelData::U16(
            (0..80 * 72)
                .flat_map(|i| {
                    [
                        (i * 37) as u16,
                        (i / 80 * 91) as u16,
                        (i * 7) as u16,
                        if i % 3 == 0 { 0 } else { 65535 },
                    ]
                })
                .collect(),
        ),
    )
    .unwrap();
    for (parallel_probes, preparation_cache_bytes, bound_finalists) in
        [(false, 0, false), (true, 8 << 20, true), (true, 0, true)]
    {
        let result = check(
            &image,
            &Options {
                threads: 4,
                probe_side: 32,
                max_probe_encodes: 12,
                candidates: vec![Candidate::Prev1, Candidate::Prev2],
                parallel_probes,
                preparation_cache_bytes,
                bound_finalists,
                ..Default::default()
            },
        );
        assert!(result.report.candidates.iter().all(|c| c.error.is_none()));
        assert!(result.report.probe_workers <= 4);
        if parallel_probes {
            assert!(result.report.probe_workers > 1);
        } else {
            assert_eq!(result.report.probe_workers, 1);
        }
        assert!(result.report.preparation_reuse.peak_bytes <= preparation_cache_bytes);
        if preparation_cache_bytes == 0 {
            assert_eq!(result.report.preparation_reuse.hits, 0);
        } else {
            assert!(result.report.preparation_reuse.hits > 0);
        }
        if !bound_finalists {
            assert!(
                result
                    .report
                    .candidates
                    .iter()
                    .all(|c| c.full_lower_bound.is_none())
            );
        }
    }
}

#[test]
fn routed_rgb_and_invisible_rgba_are_exact() {
    for threads in [1, 4] {
        for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
            let channels = layout.channels();
            let pixels = (0..96 * 80)
                .flat_map(|i| {
                    let x = i % 96;
                    let y = i / 96;
                    [
                        (x * 7) as u8,
                        (y * 3) as u8,
                        (x + y) as u8,
                        if x % 3 == 0 { 0 } else { 170 },
                    ][..channels]
                        .to_vec()
                })
                .collect();
            let image = Image::new(96, 80, layout, PixelData::U8(pixels)).unwrap();
            let result = check(
                &image,
                &Options {
                    threads,
                    probe_side: 32,
                    max_probe_encodes: 15,
                    candidates: vec![
                        Candidate::Prev1,
                        Candidate::OffsetDistribution,
                        Candidate::ShallowFull,
                    ],
                    ..Default::default()
                },
            );
            assert!(result.report.probe_encodes > 0);
            assert!(
                result.report.candidates.iter().all(|c| c.error.is_none()),
                "{:?}",
                result.report.candidates
            );
        }
    }
}

#[test]
fn all_catalog_candidates_build_or_explicitly_decline_and_decode() {
    let image = Image::new(
        19,
        17,
        PixelLayout::Rgba,
        PixelData::U16(
            (0..19 * 17)
                .flat_map(|i| {
                    [
                        (i * 199 % 65536) as u16,
                        60000,
                        (i % 17 * 900) as u16,
                        if i % 2 == 0 { 0 } else { 65535 },
                    ]
                })
                .collect(),
        ),
    )
    .unwrap();
    // Each requested plan gets a direct, small complete trial. No big corpus.
    for &candidate in portfolio::ALL {
        match candidate.build(&image, &Default::default(), 9) {
            Ok(options) => {
                // Check the candidate itself, even when routing would discard
                // it for being larger. Verifying only the winner hides bugs.
                let bytes = rs_jxl::encode(&image, &options).unwrap();
                let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
                assert!(decoded.complete, "{candidate:?}");
                assert_eq!(&decoded.frames[0].pixels, image.pixels(), "{candidate:?}");
            }
            Err(_) => assert!(
                candidate.supports(&image, 9).is_some(),
                "unexpected projection failure: {candidate:?}"
            ),
        }
        let result = check(
            &image,
            &Options {
                candidates: vec![candidate],
                max_finalists: 1,
                ..Default::default()
            },
        );
        for trial in &result.report.candidates {
            assert_ne!(
                trial.decision, "full-error",
                "{candidate:?}: {:?}",
                trial.error
            );
        }
    }
}

#[test]
fn no_probe_budget_does_not_trigger_unbounded_search() {
    let image = Image::new(64, 96, PixelLayout::Gray, PixelData::U8(vec![63; 64 * 96])).unwrap();
    let result = check(
        &image,
        &Options {
            probe_side: 32,
            max_probe_encodes: 0,
            ..Default::default()
        },
    );
    assert_eq!(result.report.full_encodes, 1);
    assert_eq!(result.report.probe_encodes, 0);
}
