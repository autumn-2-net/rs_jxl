use super::*;
use rs_jxl::modular::source_options::SourceBuffering;

fn args(flags: &[&str]) -> encode::EncodeArgs {
    let mut argv = vec!["rs-jxl", "encode", "-", "-", "--source-effort", "9"];
    argv.extend_from_slice(flags);
    expect_encode(Cli::try_parse_from(argv).unwrap().command)
}

#[test]
fn source_overrides_reach_the_typed_job_without_enabling_search() {
    let job = args(&[
        "--source-group-size-shift",
        "3",
        "--source-max-properties",
        "2",
        "--source-prev-props",
        "2",
        "--source-buffering",
        "full",
        "--source-responsive",
        "--source-precise-entropy",
        "--source-lz77-careful-first-pass",
        "--source-bucket-matcher",
        "--source-refine-entropy",
        "--source-refine-leaves",
        "--source-refine-offsets",
        "--source-refine-context-map",
    ])
    .job()
    .unwrap();
    let refinement = &job.options.modular.refinement;
    assert!(refinement.entropy && refinement.leaves);
    assert!(refinement.offsets.is_some() && refinement.context_map.is_some());
    let source = job.source.unwrap();
    assert!(matches!(
        source.search,
        rs_jxl::codec::SourceSearch::Disabled
    ));
    let params = source.params;
    assert_eq!(params.group_size_shift, Some(3));
    assert_eq!(params.max_properties, 2);
    assert_eq!(params.previous_property_subset, 2);
    assert_eq!(params.buffering, SourceBuffering::FullFrame);
    assert!(params.responsive && params.precise_entropy);
    assert!(params.lz77_careful_first_pass && params.bucket_matcher);
}

#[test]
fn source_override_flags_require_source_mode_and_validate_ranges() {
    for flags in [
        vec!["--source-group-size-shift", "2"],
        vec!["--source-max-properties", "2"],
        vec!["--source-prev-props", "1"],
        vec!["--source-buffering", "full"],
        vec!["--source-responsive"],
        vec!["--source-precise-entropy"],
        vec!["--source-refine-entropy"],
        vec!["--source-refine-leaves"],
        vec!["--source-refine-offsets"],
        vec!["--source-refine-context-map"],
        vec!["--source-lz77-careful-first-pass"],
        vec!["--source-bucket-matcher"],
    ] {
        let mut argv = vec!["rs-jxl", "encode", "-", "-"];
        argv.extend(flags);
        assert!(Cli::try_parse_from(argv).is_err());
    }
    for (flag, value) in [
        ("--source-group-size-shift", "4"),
        ("--source-max-properties", "17"),
        ("--source-prev-props", "0"),
        ("--source-prev-props", "3"),
        ("--source-buffering", "invalid"),
    ] {
        assert!(
            Cli::try_parse_from([
                "rs-jxl",
                "encode",
                "-",
                "-",
                "--source-effort",
                "9",
                flag,
                value,
            ])
            .is_err()
        );
    }
}

#[test]
fn previous_property_subsets_reach_main_and_reference_sampling() {
    use rs_jxl::scheduler::adapter::{project_native, source_patch_reference_policy};
    use rs_jxl::scheduler::{CandidateParams, NativeResourcePolicy};
    use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout};

    let image = Image::new(
        16,
        16,
        PixelLayout::Rgb,
        PixelData::U8(vec![7; 16 * 16 * 3]),
    )
    .unwrap();
    let base = EncodeOptions::default();
    for effort in [8, 9, 10] {
        for subset in [1, 2, 4] {
            let params = CandidateParams {
                max_properties: 2,
                previous_property_subset: subset,
                ..CandidateParams::default().backend(effort).unwrap()
            };
            let expected = match (effort, subset) {
                (8, _) | (_, 1) => vec![19, 23],
                (_, 2) => vec![17, 19, 21, 23],
                _ => (16..24).collect(),
            };
            let projected = project_native(&params, &image, &base).unwrap();
            let options = projected.options.unwrap();
            let sampling = options.modular.sampling.unwrap();
            assert_eq!(
                sampling
                    .properties
                    .into_iter()
                    .filter(|&p| p >= 16)
                    .collect::<Vec<_>>(),
                expected
            );
            let reference = source_patch_reference_policy(
                &params,
                &image,
                &base,
                NativeResourcePolicy::default(),
            )
            .unwrap()
            .resolve(&image)
            .unwrap();
            let props = reference.modular.sampling.unwrap().properties;
            assert_eq!(
                props.into_iter().filter(|&p| p >= 16).collect::<Vec<_>>(),
                expected
            );
            let identity = CandidateParams {
                previous_property_subset: if subset == 4 { 1 } else { 4 },
                ..params.clone()
            };
            assert_ne!(params, identity);
        }
    }
}

#[test]
fn each_buffering_mode_has_an_explicit_mapping() {
    for (name, expected) in [
        ("auto", SourceBuffering::Auto),
        ("full", SourceBuffering::FullFrame),
        ("above2048", SourceBuffering::StreamingAbove2048),
        ("streaming", SourceBuffering::Streaming),
    ] {
        assert_eq!(
            args(&["--source-buffering", name])
                .job()
                .unwrap()
                .source
                .unwrap()
                .params
                .buffering,
            expected
        );
    }
}
