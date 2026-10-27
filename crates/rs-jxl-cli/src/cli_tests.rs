use super::*;
use crate::encode::encode_image;
use clap::CommandFactory;
use rs_jxl::codec::SourceSearch;
use rs_jxl::scheduler::{CandidateParams, NativeAcceptance, pilot::PilotOptions};

fn encode_args(extra: &[&str]) -> encode::EncodeArgs {
    let mut argv = vec!["rs-jxl", "encode", "-", "-"];
    argv.extend_from_slice(extra);
    expect_encode(Cli::try_parse_from(argv).unwrap().command)
}

#[test]
fn cli_shape_and_defaults() {
    Cli::command().debug_assert();
    let cli = Cli::try_parse_from(["rs-jxl", "encode", "-", "-"]).unwrap();
    let args = expect_encode(cli.command);
    assert_eq!(args.threads, 1);
    assert!(!args.force);
    assert!(!args.strip_metadata);
    assert_eq!(args.source_effort, None);
    assert!(!args.source_search);
    assert!(!args.source_scout);
    assert_eq!(args.source_sample_limit, None);
}

#[test]
fn source_arguments_and_resources() {
    for effort in 1..=11 {
        let cli = Cli::try_parse_from([
            "rs-jxl",
            "encode",
            "a",
            "b",
            "--source-effort",
            &effort.to_string(),
        ])
        .unwrap();
        let args = expect_encode(cli.command);
        assert_eq!(args.source_effort, Some(effort));
        assert!(!args.source_search);
        assert_eq!(args.source_sample_limit, None);
    }
    let cli = Cli::try_parse_from([
        "rs-jxl",
        "encode",
        "a",
        "b",
        "--source-effort",
        "3",
        "--source-search",
        "--source-sample-limit",
        "1048576",
        "--threads",
        "2",
        "--strip-metadata",
        "--force",
    ])
    .unwrap();
    let args = expect_encode(cli.command);
    assert!(args.source_search);
    assert_eq!(args.source_sample_limit, Some(1_048_576));
    assert_eq!(args.threads, 2);
    for effort in ["0", "12", "-1", "abc", "256"] {
        assert!(
            Cli::try_parse_from(["rs-jxl", "encode", "a", "b", "--source-effort", effort,])
                .is_err()
        );
    }
    for limit in ["0", "-1", "abc", "18446744073709551616"] {
        assert!(
            Cli::try_parse_from([
                "rs-jxl",
                "encode",
                "a",
                "b",
                "--source-effort",
                "3",
                "--source-sample-limit",
                limit,
            ])
            .is_err()
        );
    }
    for extra in [
        vec!["--source-search"],
        vec!["--source-scout"],
        vec!["--source-pilot"],
        vec!["--source-sample-limit", "1048576"],
    ] {
        let mut argv = vec!["rs-jxl", "encode", "a", "b"];
        argv.extend(extra);
        assert!(Cli::try_parse_from(argv).is_err());
    }
}

#[test]
fn typed_source_jobs_keep_candidate_and_search_requests_distinct() {
    for effort in 1..=11 {
        let effort_arg = effort.to_string();
        for search in [false, true] {
            let mut flags = vec!["--source-effort", &effort_arg];
            if search {
                flags.push("--source-search");
            }
            let job = encode_args(&flags).job().unwrap();
            assert_eq!(job.options.threads, 1);
            assert_eq!(job.options.modular.sample_limit, None);
            assert_eq!(job.options.modular.ma_budget_mode, None);
            assert_eq!(job.options.modular.ma_max_leaves, None);
            let source = job.source.unwrap();
            assert_eq!(
                source.params,
                CandidateParams::default().backend(effort).unwrap()
            );
            if search {
                // Effort 11 is still Auto here; the codec owns tournament selection.
                assert!(matches!(source.search, SourceSearch::Auto));
            } else {
                assert!(matches!(source.search, SourceSearch::Disabled));
            }
            assert_eq!(source.backend.acceptance, NativeAcceptance::RequireExact);
            assert_eq!(source.backend.preparation_cache_bytes, 0);
            assert!(!source.backend.patches_off);
            assert!(!source.backend.probe_palette);
            assert!(!source.backend.literal_palette_tools);
            assert!(!source.backend.routed_tools);
        }
    }
}

#[test]
fn typed_source_jobs_keep_scout_and_pilot_options_explicit() {
    for effort in ["9", "10"] {
        let base = ["--source-effort", effort, "--source-search"];
        let mut flags = base.to_vec();
        flags.push("--source-scout");
        let source = encode_args(&flags).job().unwrap().source.unwrap();
        assert!(matches!(source.search, SourceSearch::Scout));
        for progressive in [false, true] {
            let mut flags = base.to_vec();
            flags.push("--source-pilot");
            if progressive {
                flags.push("--source-progressive-groups");
            }
            let source = encode_args(&flags).job().unwrap().source.unwrap();
            let SourceSearch::Pilot(options) = source.search else {
                panic!("expected pilot request");
            };
            assert_eq!(
                options,
                PilotOptions {
                    group_race: progressive.then(Default::default),
                    ..Default::default()
                }
            );
        }
    }
}

#[test]
fn typed_jobs_forward_shared_resource_controls_without_changing_modes() {
    use rs_jxl::modular::ma_train::MaBudgetMode;

    let routes: &[&[&str]] = &[
        &[],
        &["--source-effort", "3"],
        &["--source-effort", "11"],
        &["--source-effort", "3", "--source-search"],
        &["--source-effort", "11", "--source-search"],
        &["--source-effort", "9", "--source-search", "--source-scout"],
        &[
            "--source-effort",
            "10",
            "--source-search",
            "--source-pilot",
            "--source-progressive-groups",
        ],
    ];
    for route in routes {
        for (mode, expected) in [
            ("strict", MaBudgetMode::Strict),
            ("bounded", MaBudgetMode::Bounded),
        ] {
            let mut flags = route.to_vec();
            flags.extend_from_slice(&[
                "--threads",
                "2",
                "--ma-budget-mode",
                mode,
                "--ma-max-leaves",
                "17",
                "--ma-prune",
                "--strip-metadata",
                "--force",
            ]);
            if !route.is_empty() {
                flags.extend_from_slice(&["--source-sample-limit", "4096"]);
            }
            let args = encode_args(&flags);
            assert!(args.strip_metadata && args.force);
            let job = args.job().unwrap();
            assert_eq!(job.source.is_some(), !route.is_empty());
            assert_eq!(job.options.threads, 2);
            let modular = &job.options.modular;
            assert_eq!(modular.ma_budget_mode, Some(expected));
            assert_eq!(modular.ma_max_leaves, Some(17));
            assert!(modular.source_property_dedup && modular.source_ma_dominance);
            assert_eq!(modular.sample_limit, (!route.is_empty()).then_some(4096));
        }
    }
}

#[test]
fn typed_manual_jobs_keep_tool_defaults_overrides_and_validation() {
    use rs_jxl::modular::predictor::Predictor;

    let job = encode_args(&[]).job().unwrap();
    assert!(job.source.is_none());
    assert_eq!(job.options.threads, 1);
    assert_eq!(job.options.modular.rct, Some(6));
    assert_eq!(job.options.modular.tree_samples, 1024);
    assert_eq!(job.options.modular.max_tree_leaves, 16);
    assert_eq!(job.options.modular.ma_budget_mode, None);
    assert_eq!(job.options.modular.sample_limit, None);

    let job = encode_args(&[
        "--rct",
        "0",
        "--predictor",
        "0",
        "--weighted-preset",
        "4",
        "--tree-samples",
        "2048",
        "--tree-leaves",
        "32",
        "--tree-split-threshold",
        "24",
        "--previous-channels",
        "2",
        "--squeeze",
        "horizontal,vertical",
        "--lz77",
        "off",
        "--group-size-shift",
        "2",
        "--shared-tree",
        "--source-patches",
        "--progressive",
    ])
    .job()
    .unwrap();
    assert!(job.source.is_none());
    let options = job.options;
    assert_eq!(options.modular.rct, Some(0));
    assert_eq!(options.modular.predictor, Predictor::Zero);
    assert_eq!(options.modular.weighted_preset, 4);
    assert_eq!(options.modular.tree_samples, 2048);
    assert_eq!(options.modular.max_tree_leaves, 32);
    assert_eq!(options.modular.split_threshold, 24.0);
    assert_eq!(options.modular.previous_channels, 2);
    assert_eq!(options.modular.squeeze, [true, false]);
    assert!(!options.modular.lz77);
    assert_eq!(options.group_size_shift, 2);
    assert!(options.shared_tree);
    assert!(options.source_patches.is_some());
    assert_eq!(
        options.passes,
        rs_jxl::modular::layout::Passes::progressive()
    );
    for flag in ["--tree-samples", "--tree-leaves"] {
        assert!(encode_args(&[flag, "0"]).job().is_err());
    }
}

#[test]
fn source_scout_is_explicit_and_rejects_unsupported_efforts() {
    use rs_jxl::{Image, PixelData, PixelLayout};
    let image = Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![0])).unwrap();
    for effort in ["8", "11"] {
        let cli = Cli::try_parse_from([
            "rs-jxl",
            "encode",
            "a",
            "b",
            "--source-effort",
            effort,
            "--source-search",
            "--source-scout",
        ])
        .unwrap();
        assert!(
            encode_image(&image, &expect_encode(cli.command))
                .unwrap_err()
                .to_string()
                .contains("9 or 10")
        );
    }
    let cli = Cli::try_parse_from([
        "rs-jxl",
        "encode",
        "a",
        "b",
        "--source-effort",
        "9",
        "--source-search",
        "--source-scout",
    ])
    .unwrap();
    assert!(expect_encode(cli.command).source_scout);
}

#[test]
fn content_pilot_is_explicit_and_excludes_other_search_presets() {
    let image = rs_jxl::Image::new(
        1,
        1,
        rs_jxl::PixelLayout::Gray,
        rs_jxl::PixelData::U8(vec![0]),
    )
    .unwrap();
    for effort in ["8", "11"] {
        let cli = Cli::try_parse_from([
            "rs-jxl",
            "encode",
            "a",
            "b",
            "--source-effort",
            effort,
            "--source-search",
            "--source-pilot",
        ])
        .unwrap();
        assert!(
            encode_image(&image, &expect_encode(cli.command))
                .unwrap_err()
                .to_string()
                .contains("9 or 10")
        );
    }
    assert!(
        Cli::try_parse_from([
            "rs-jxl",
            "encode",
            "a",
            "b",
            "--source-effort",
            "9",
            "--source-search",
            "--source-pilot",
            "--source-scout"
        ])
        .is_err()
    );
    let cli = Cli::try_parse_from([
        "rs-jxl",
        "encode",
        "a",
        "b",
        "--source-effort",
        "9",
        "--source-search",
        "--source-pilot",
    ])
    .unwrap();
    assert!(expect_encode(cli.command).source_pilot);
    let cli = Cli::try_parse_from([
        "rs-jxl",
        "encode",
        "a",
        "b",
        "--source-effort",
        "9",
        "--source-search",
        "--source-pilot",
        "--source-progressive-groups",
    ])
    .unwrap();
    assert!(expect_encode(cli.command).source_progressive_groups);
    assert!(
        Cli::try_parse_from(["rs-jxl", "encode", "a", "b", "--source-progressive-groups",])
            .is_err()
    );
}

#[test]
fn explicit_effort_eleven_maps_to_tectonic_not_effort_ten() {
    let cli = Cli::try_parse_from(["rs-jxl", "encode", "a", "b", "--source-effort", "11"]).unwrap();
    let args = expect_encode(cli.command);
    let source = args.job().unwrap().source.unwrap();
    let candidate = source.params;
    assert_eq!(candidate.speed_tier, -1);
    assert!(!args.source_search);
    assert!(matches!(source.search, SourceSearch::Disabled));
    assert_ne!(candidate, CandidateParams::default().backend(10).unwrap());
    assert_eq!(candidate.training_policy(3, 22).node_threshold, 61);
}

#[test]
fn source_rejects_every_explicit_manual_tool_including_defaults() {
    let cases: &[&[&str]] = &[
        &["--rct", "6"],
        &["--predictor", "14"],
        &["--weighted-preset", "0"],
        &["--rct-candidates", "1"],
        &["--weighted-candidates", "1"],
        &["--palette-colors", "256"],
        &["--palette-colors", "256", "--palette-predictor", "0"],
        &["--squeeze", "horizontal"],
        &["--entropy", "auto"],
        &["--lz77", "lazy"],
        &["--tree-samples", "1024"],
        &["--tree-leaves", "16"],
        &["--tree-split-threshold", "48"],
        &["--previous-channels", "0"],
        &["--group-size-shift", "1"],
        &["--shared-tree"],
        &["--patches"],
        &["--global-palette", "--palette-colors", "256"],
        &["--responsive"],
        &["--progressive"],
        &["--source-patches"],
    ];
    for &manual in cases {
        // The same tool arguments remain legal without source mode.
        let mut argv = vec!["rs-jxl", "encode", "a", "b"];
        argv.extend_from_slice(manual);
        Cli::try_parse_from(&argv).unwrap();
        for source_first in [false, true] {
            for search in [false, true] {
                let mut argv = vec!["rs-jxl", "encode", "a", "b"];
                let mut source = vec!["--source-effort", "3"];
                if search {
                    source.push("--source-search");
                }
                if source_first {
                    argv.extend_from_slice(&source);
                    argv.extend_from_slice(manual);
                } else {
                    argv.extend_from_slice(manual);
                    argv.extend_from_slice(&source);
                }
                let error = Cli::try_parse_from(argv).unwrap_err();
                assert_eq!(
                    error.kind(),
                    clap::error::ErrorKind::ArgumentConflict,
                    "{manual:?}"
                );
            }
        }
    }
}

#[test]
fn only_supported_options_are_accepted() {
    for option in ["--effort", "--quality", "--lossy"] {
        assert!(Cli::try_parse_from(["rs-jxl", "encode", "a", "b", option, "1"]).is_err());
    }
    for count in ["0", "-1", "257", "abc", "18446744073709551616"] {
        assert!(Cli::try_parse_from(["rs-jxl", "encode", "a", "b", "--threads", count]).is_err());
    }
    assert!(Cli::try_parse_from(["rs-jxl", "decode", "a", "b"]).is_ok());
}
