use super::*;
use rs_jxl::modular::ma_train::MaBudgetMode;

#[test]
fn ma_budget_lab_parser_defaults_and_content_pilot_propagation() {
    let args = Args::try_parse_from(["encoder_lab", "input.png"]).unwrap();
    let mut options = rs_jxl::ModularOptions::default();
    args.ma_budget.apply(&mut options);
    args.ma_pruning.apply(&mut options);
    assert!(!options.source_ma_dominance && !options.source_property_dedup);
    assert_eq!(options.ma_budget_mode, None);
    assert_eq!(options.ma_max_leaves, None);
    for (value, mode) in [
        ("strict", MaBudgetMode::Strict),
        ("bounded", MaBudgetMode::Bounded),
    ] {
        let args = Args::try_parse_from([
            "encoder_lab",
            "input.png",
            "--search",
            "--content-pilot",
            "--subtrees",
            "--ma-prune",
            "--ma-budget-mode",
            value,
            "--ma-max-leaves",
            "3",
        ])
        .unwrap();
        args.ma_budget.apply(&mut options);
        args.ma_pruning.apply(&mut options);
        assert!(options.source_ma_dominance && options.source_property_dedup);
        assert_eq!(options.ma_budget_mode, Some(mode));
        assert_eq!(options.ma_max_leaves, Some(3));
        assert!(args.content_pilot && args.subtrees);
    }
}

#[test]
fn ma_budget_lab_parser_rejects_invalid_values() {
    for (flag, value) in [
        ("--ma-budget-mode", "auto"),
        ("--ma-max-leaves", "0"),
        ("--ma-max-leaves", "2097153"),
        ("--ma-max-leaves", "-1"),
        ("--ma-max-leaves", "many"),
    ] {
        assert!(Args::try_parse_from(["encoder_lab", "input.png", flag, value]).is_err());
    }
}
