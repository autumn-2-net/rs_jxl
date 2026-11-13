//! Fixed-input, single-decision Palette counterfactuals; not a search preset.
use clap::{Parser, ValueEnum};
use rs_jxl::{
    EncodeOptions, Error, Image,
    modular::palette::{
        PaletteMode, PaletteOutcome, PaletteStage,
        admission::{self, PlaneScore, Score},
        counterfactual::{Choice, Experiment, Record, Target},
    },
    scheduler::{CandidateParams, adapter},
};
use serde_json::{Value, json};
use std::{fs, io::Write, path::PathBuf, sync::Arc, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Absolute,
    ExactDeltaGradient,
    ExactDeltaLeft,
    ExactDeltaTop,
    ExactZeroGradient,
    ExactZeroLeft,
    ExactZeroTop,
}

impl Mode {
    fn native(self) -> PaletteMode {
        PaletteMode::from_source_mode(self as u8).expect("seven source modes")
    }
    fn name(self) -> &'static str {
        match self {
            Self::Absolute => "absolute",
            Self::ExactDeltaGradient => "exact-delta-gradient",
            Self::ExactDeltaLeft => "exact-delta-left",
            Self::ExactDeltaTop => "exact-delta-top",
            Self::ExactZeroGradient => "exact-zero-gradient",
            Self::ExactZeroLeft => "exact-zero-left",
            Self::ExactZeroTop => "exact-zero-top",
        }
    }
}

fn mode_name(mode: PaletteMode) -> &'static str {
    for m in Mode::value_variants() {
        if m.native() == mode {
            return m.name();
        }
    }
    "unsupported"
}

fn parse_choice(value: &str) -> std::result::Result<String, String> {
    if value == "skip" || Mode::from_str(value, false).is_ok() {
        Ok(value.to_owned())
    } else {
        Err("expected skip or a standard palette mode".into())
    }
}

fn choice(value: &str) -> Choice {
    if value == "skip" {
        Choice::Skip
    } else {
        Choice::Palette(
            Mode::from_str(value, false)
                .expect("validated choice")
                .native(),
        )
    }
}

fn choice_name(choice: Choice) -> &'static str {
    match choice {
        Choice::Keep => "keep",
        Choice::Skip => "skip",
        Choice::Palette(mode) => mode_name(mode),
    }
}

fn parse_target(value: &str) -> std::result::Result<Target, String> {
    let parts: Vec<_> = value.split(':').collect();
    if parts.len() != 3 {
        return Err("target must be global:STREAM:ORDINAL or local:STREAM:ORDINAL".into());
    }
    let stage = match parts[0] {
        "global" => PaletteStage::PreRct,
        "local" => PaletteStage::PostRct,
        _ => return Err("invalid palette stage".into()),
    };
    Ok(Target {
        stage,
        stream_id: parts[1].parse().map_err(|_| "invalid stream ID")?,
        ordinal: parts[2].parse().map_err(|_| "invalid attempt ordinal")?,
    })
}

fn stage_name(stage: PaletteStage) -> &'static str {
    match stage {
        PaletteStage::PreRct => "global",
        PaletteStage::PostRct => "local",
    }
}

fn target_json(target: Target) -> Value {
    json!({"stage":stage_name(target.stage), "stream_id":target.stream_id, "ordinal":target.ordinal})
}

fn event_json(record: &Record) -> Value {
    let attempt = record.attempt.as_ref().map(|a| {
        json!({
            "mode":mode_name(a.mode), "width":a.width, "height":a.height,
            "begin":a.begin, "channels":a.channels, "limit":a.limit,
            "colors":a.colors, "deltas":a.deltas, "delta_pixels":a.delta_pixels,
            "predictor":format!("{:?}",a.predictor), "cost_before":a.cost_before,
            "cost_after":a.cost_after, "cost_bypassed":a.cost_bypassed,
            "exact_delta_attempted":a.kernel.exact_delta_attempted,
            "exact_delta_produced":a.kernel.exact_delta_produced,
            "exact_zero_attempted":a.kernel.exact_zero_attempted,
            "zero_mask_eligible":a.kernel.zero_mask_eligible,
            "zero_mask_matches":a.kernel.zero_mask_matches,
            "absolute_attempted":a.kernel.absolute_attempted,
            "absolute_produced":a.kernel.absolute_produced,
            "outcome":match a.outcome {
                PaletteOutcome::Accepted => "accepted",
                PaletteOutcome::CostRejected => "cost_rejected",
                PaletteOutcome::KernelDeclined => "kernel_declined",
            },
        })
    });
    let mut result = target_json(record.target);
    result["choice"] = json!(choice_name(record.choice));
    result["cost_forced"] = json!(matches!(record.choice, Choice::Palette(_)));
    result["unavailable"] = json!(record.unavailable);
    result["attempt"] = json!(attempt);
    result
}

fn score_json(score: &Score) -> Value {
    let planes: Vec<_> = score.planes.iter().map(|p: &PlaneScore| json!({
        "population":p.population, "sampled":p.sampled, "training_sites":p.training_sites,
        "leaves":p.leaves, "tree_bits":p.tree_bits,
        "literal_header_bits":p.literal_header_bits, "literal_data_bits":p.literal_data_bits,
        "lz_header_bits":p.lz_header_bits, "lz_data_bits":p.lz_data_bits,
        "lz_matches":p.lz_matches, "lz_matched_samples":p.lz_matched_samples,
        "used_lz":p.used_lz, "total_bits":p.total_bits,
    })).collect();
    json!({"total_bits":score.total_bits,"transform_header_bits":score.transform_header_bits,"planes":planes})
}

fn admission_json(record: &admission::Record) -> Value {
    let mut result = target_json(record.target);
    result["mode"] = json!(mode_name(record.mode));
    result["skip"] = record.skip.as_ref().map_or(Value::Null, score_json);
    result["candidate"] = record.candidate.as_ref().map_or(Value::Null, score_json);
    result["old_would_accept"] = json!(record.old_would_accept);
    result["proxy_would_accept"] = json!(record.proxy_would_accept);
    result["vetoed"] = json!(record.vetoed);
    result["forced"] = json!(record.forced);
    result["unavailable"] = json!(record.unavailable);
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Admission {
    Shadow,
    Veto,
}

impl Admission {
    fn native(self) -> admission::Mode {
        match self {
            Self::Shadow => admission::Mode::Shadow,
            Self::Veto => admission::Mode::Veto,
        }
    }
}

#[derive(Parser)]
struct Args {
    input: PathBuf,
    /// Fresh output directory. All successful trials produce complete JXL files.
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long, value_enum, default_value = "absolute")]
    baseline_mode: Mode,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    /// Omit for discovery. Ordinal is local to one stage/stream, zero-based.
    #[arg(long, value_parser = parse_target)]
    target: Option<Target>,
    #[arg(long, num_args = 1.., value_parser = parse_choice,
          default_values = ["skip", "absolute", "exact-delta-gradient"])]
    choices: Vec<String>,
    /// Omit to disable. With --target, score/veto only that frozen decision.
    /// Forced counterfactual branches always take precedence over veto mode.
    #[arg(long, value_enum)]
    admission: Option<Admission>,
}

fn encode_trial(
    args: &Args,
    image: &Image,
    options: &EncodeOptions,
    experiment: Arc<Experiment>,
    name: &str,
) -> Result<Value> {
    let mut options = options.clone();
    let policy = options.modular.palette_policy.as_mut().unwrap();
    policy.counterfactual = Some(Arc::clone(&experiment));
    let admission = args
        .admission
        .map(|mode| Arc::new(admission::Experiment::new(mode.native(), args.target)));
    policy.admission = admission.clone();
    let start = Instant::now();
    let result = rs_jxl::encode(image, &options);
    let seconds = start.elapsed().as_secs_f64();
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("invalid native encode duration".into());
    }
    let pixels = f64::from(image.width()) * f64::from(image.height());
    let mut records = experiment.events();
    records.sort_by_key(|r| {
        (
            r.target.stage == PaletteStage::PostRct,
            r.target.stream_id,
            r.target.ordinal,
        )
    });
    let events: Vec<_> = records.iter().map(event_json).collect();
    let mut admission_events = admission.as_ref().map_or_else(Vec::new, |e| e.events());
    admission_events.sort_by_key(|r| {
        (
            r.target.stage == PaletteStage::PostRct,
            r.target.stream_id,
            r.target.ordinal,
        )
    });
    let admission_events: Vec<_> = admission_events.iter().map(admission_json).collect();
    let admission_seconds = admission.as_ref().map_or(0.0, |e| e.scoring_seconds());
    let mut report = json!({
        "name":name, "seconds":seconds, "events":events,
        "admission_seconds":admission_seconds,
        "admission_timing":"summed elapsed scoring comparisons; concurrent groups may overlap; not wall time",
        "admission_events":admission_events,
        "fixed_input_verified":args.target.is_some(),
    });
    match result {
        Ok(bytes) => {
            experiment.validate_complete()?;
            let filename = format!("{name}.jxl");
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(args.out_dir.join(&filename))?
                .write_all(&bytes)?;
            report["status"] = json!("ok");
            report["file"] = json!(filename);
            report["bytes"] = json!(bytes.len());
            report["bpp"] = json!(8.0 * bytes.len() as f64 / pixels);
            report["mpix_per_second"] = json!(pixels / 1e6 / seconds);
        }
        Err(Error::Encoding(ref message))
            if message == "counterfactual palette kernel unavailable at fixed input"
                && experiment.unavailable() =>
        {
            experiment.validate_complete()?;
            report["status"] = json!("unavailable");
            for key in ["file", "bytes", "bpp", "mpix_per_second"] {
                report[key] = Value::Null;
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(report)
}

// Compare only forced, available alternatives. The baseline need not have
// admitted its candidate, so its output size is not a candidate-size label.
fn fixed_state_validation(trials: &[Value], target: Target) -> Result<Value> {
    let target = target_json(target);
    let selected = |event: &&Value| {
        event["stage"] == target["stage"]
            && event["stream_id"] == target["stream_id"]
            && event["ordinal"] == target["ordinal"]
    };
    let mut skip_bits = None;
    for trial in trials {
        if let Some(events) = trial["admission_events"].as_array() {
            for event in events.iter().filter(selected) {
                if let Some(bits) = event["skip"]["total_bits"].as_f64() {
                    if skip_bits.is_some_and(|previous| previous != bits) {
                        return Err("pilot Skip costs differ at the verified frozen input".into());
                    }
                    skip_bits = Some(bits);
                }
            }
        }
    }
    let mut rankings = Vec::new();
    for trial in trials {
        if trial["name"] == "baseline" || trial["status"] != "ok" {
            continue;
        }
        let bits = if trial["name"] == "skip" {
            skip_bits
        } else {
            trial["admission_events"]
                .as_array()
                .and_then(|events| events.iter().find(selected))
                .and_then(|event| event["candidate"]["total_bits"].as_f64())
        };
        rankings.push(json!({"choice":trial["name"],"bytes":trial["bytes"],"proxy_bits":bits}));
    }
    rankings.sort_by(|a, b| {
        a["proxy_bits"]
            .as_f64()
            .unwrap_or(f64::INFINITY)
            .total_cmp(&b["proxy_bits"].as_f64().unwrap_or(f64::INFINITY))
    });
    let measured_best = rankings.iter().filter_map(|r| r["bytes"].as_u64()).min();
    let regret = rankings
        .first()
        .filter(|r| !r["proxy_bits"].is_null())
        .and_then(|r| r["bytes"].as_u64())
        .zip(measured_best)
        .map(|(predicted, best)| predicted - best);
    Ok(json!({
        "rankings":rankings,"proxy_winner_regret_bytes":regret,
        "scope":"only available requested forced alternatives; complete-output labels, not isolated stream costs",
        "limitations":"four-leaf stateless MA, independent plane models, fixed420 prefix, bounded window-local LZ; no gain claim",
    }))
}

fn run(args: Args) -> Result<()> {
    let mut names = std::collections::HashSet::new();
    if args.choices.iter().any(|name| !names.insert(name)) {
        return Err("duplicate counterfactual choice".into());
    }
    fs::create_dir(&args.out_dir)?;
    let image = rs_jxl_image::decode(&fs::read(&args.input)?, true)?.image;
    let params = CandidateParams::default().backend(9)?;
    let base = EncodeOptions {
        threads: usize::from(args.threads),
        ..Default::default()
    };
    let mut options = adapter::project_native(&params, &image, &base)?.exact_options()?;
    options.patches = None;
    options.source_patches = None;
    options.source_patch_reference = None;
    options.tile_patches = None;
    options.modular.candidate_cache = None;
    if options.modular.fast_context.is_some() || options.modular.block_context.is_some() {
        return Err("palette admission experiment requires ordinary native MA".into());
    }
    let policy = options
        .modular
        .palette_policy
        .as_mut()
        .ok_or("missing palette policy")?;
    policy.mode = args.baseline_mode.native();
    let configured = json!({"palette_colors":policy.palette_colors,"groupwise":policy.groupwise});
    let experiment = Arc::new(
        args.target
            .map_or_else(Experiment::observe, Experiment::capture),
    );
    let mut trials = vec![encode_trial(
        &args,
        &image,
        &options,
        Arc::clone(&experiment),
        "baseline",
    )?];
    if args.target.is_some() {
        let snapshot = experiment.snapshot().ok_or("target snapshot missing")?;
        for name in &args.choices {
            let replay = Arc::new(Experiment::replay(Arc::clone(&snapshot), choice(name)));
            trials.push(encode_trial(&args, &image, &options, replay, name)?);
        }
    }
    let validation = if args.admission.is_some() {
        args.target
            .map(|target| fixed_state_validation(&trials, target))
            .transpose()?
    } else {
        None
    };
    println!(
        "{}",
        json!({
            "schema":1, "width":image.width(), "height":image.height(), "bits":image.bits(),
            "effort":9,"threads":args.threads,"baseline_mode":args.baseline_mode.name(),
            "target":args.target.map(target_json),"configured":configured,"trials":trials,
            "search":false,"patches_cleared":true,
            "admission":args.admission.map(|mode| format!("{mode:?}").to_lowercase()),
            "fixed_state_validation":validation,
            "timing":"native complete encode including experiment receipts and fixed-state checks; no IO/decoding",
            "continuation":"only one decision overridden; later normal policy decisions and MA/LZ77/ANS may respond",
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_and_choices_are_explicit() {
        assert_eq!(
            parse_target("local:23:2").unwrap(),
            Target {
                stage: PaletteStage::PostRct,
                stream_id: 23,
                ordinal: 2,
            }
        );
        for bad in ["local:1", "local:-1:0", "worker:1:0", "global:0:0:1"] {
            assert!(parse_target(bad).is_err());
        }
        assert!(parse_choice("../output").is_err());
        assert!(parse_choice("skip").is_ok());
        for mode in Mode::value_variants() {
            assert_eq!(mode_name(mode.native()), mode.name());
            assert_eq!(choice(mode.name()), Choice::Palette(mode.native()));
        }
    }

    #[test]
    fn downstream_scoring_is_opt_in_and_independent_of_forced_choices() {
        let base = ["palette_admission_lab", "image.png", "--out-dir", "fresh"];
        assert!(Args::try_parse_from(base).unwrap().admission.is_none());
        for mode in ["shadow", "veto"] {
            let mut arguments = base.to_vec();
            arguments.extend([
                "--admission",
                mode,
                "--target",
                "local:23:0",
                "--choices",
                "skip",
                "absolute",
                "exact-delta-top",
                "exact-zero-gradient",
            ]);
            let args = Args::try_parse_from(arguments).unwrap();
            assert!(args.admission.is_some());
            assert_eq!(args.choices.len(), 4);
            assert_eq!(args.target.unwrap().stream_id, 23);
        }
    }

    #[test]
    fn fixed_state_report_exposes_misranking_and_excludes_unavailable_bytes() {
        let target = parse_target("local:23:0").unwrap();
        let event = |bits: f64| {
            let mut event = target_json(target);
            event["skip"] = json!({"total_bits":100.0});
            event["candidate"] = json!({"total_bits":bits});
            event
        };
        let trials = vec![
            json!({"name":"baseline","status":"ok","bytes":1065338,"admission_events":[event(50.0)]}),
            json!({"name":"skip","status":"ok","bytes":1058948,"admission_events":[]}),
            json!({"name":"exact-delta-top","status":"ok","bytes":1065338,"admission_events":[event(50.0)]}),
            json!({"name":"absolute","status":"unavailable","bytes":null,"admission_events":[]}),
        ];
        let report = fixed_state_validation(&trials, target).unwrap();
        assert_eq!(report["rankings"].as_array().unwrap().len(), 2);
        assert_eq!(report["rankings"][0]["choice"], "exact-delta-top");
        assert_eq!(report["proxy_winner_regret_bytes"], 6390);
        let mut mismatched = trials;
        mismatched[2]["admission_events"][0]["skip"]["total_bits"] = json!(101.0);
        assert!(fixed_state_validation(&mismatched, target).is_err());
    }
}
