use super::*;
use crate::encode::encode_image;
use clap::CommandFactory;
use rs_jxl::scheduler::CandidateParams;

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
    let candidate = CandidateParams::default()
        .backend(args.source_effort.unwrap())
        .unwrap();
    assert_eq!(candidate.speed_tier, -1);
    assert!(!args.source_search);
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
