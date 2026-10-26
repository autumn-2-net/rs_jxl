use super::*;
use rs_jxl::EncodeOptions;
use rs_jxl::modular::ma_train::MaBudgetMode;

#[test]
fn ma_budget_parser_keeps_defaults_and_accepts_manual_source_and_pilot() {
    for route in [
        vec![],
        vec!["--source-effort", "9"],
        vec!["--source-effort", "9", "--source-search", "--source-pilot"],
    ] {
        let mut argv = vec!["rs-jxl", "encode", "a.png", "b.jxl"];
        argv.extend(route);
        let args = expect_encode(Cli::try_parse_from(&argv).unwrap().command);
        let mut options = rs_jxl::ModularOptions::default();
        args.ma_budget.apply(&mut options);
        args.ma_pruning.apply(&mut options);
        assert!(!options.source_ma_dominance && !options.source_property_dedup);
        assert_eq!(options.ma_budget_mode, None);
        assert_eq!(options.ma_max_leaves, None);
        for (flag, mode) in [
            ("strict", MaBudgetMode::Strict),
            ("bounded", MaBudgetMode::Bounded),
        ] {
            let mut explicit = argv.clone();
            explicit.extend([
                "--ma-budget-mode",
                flag,
                "--ma-max-leaves",
                "1",
                "--ma-prune",
            ]);
            let args = expect_encode(Cli::try_parse_from(explicit).unwrap().command);
            let mut options = EncodeOptions::default();
            if args.source_effort.is_none() {
                args.tools.apply(&mut options).unwrap();
            }
            args.ma_budget.apply(&mut options.modular);
            args.ma_pruning.apply(&mut options.modular);
            assert!(options.modular.source_ma_dominance && options.modular.source_property_dedup);
            assert_eq!(options.modular.ma_budget_mode, Some(mode));
            assert_eq!(options.modular.ma_max_leaves, Some(1));
        }
    }
}

#[test]
fn ma_budget_parser_rejects_invalid_modes_caps_and_retains_manual_conflicts() {
    for (flag, values) in [
        ("--ma-budget-mode", vec!["", "auto", "silent", "STRICT"]),
        (
            "--ma-max-leaves",
            vec!["0", "-1", "2097153", "1.5", "abc", "18446744073709551616"],
        ),
    ] {
        for value in values {
            assert!(Cli::try_parse_from(["rs-jxl", "encode", "a", "b", flag, value]).is_err());
        }
    }
    for cap in ["1", "2097152"] {
        assert!(
            Cli::try_parse_from(["rs-jxl", "encode", "a", "b", "--ma-max-leaves", cap]).is_ok()
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
            "--tree-leaves",
            "16",
            "--ma-budget-mode",
            "bounded",
        ])
        .is_err()
    );
}
