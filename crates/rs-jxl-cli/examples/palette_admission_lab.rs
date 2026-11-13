//! Fixed-input, single-decision Palette counterfactuals; not a search preset.
use clap::{Parser, ValueEnum};
use rs_jxl::{
    EncodeOptions, Error, Image,
    modular::palette::{
        PaletteMode, PaletteOutcome, PaletteStage,
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
}

fn encode_trial(
    args: &Args,
    image: &Image,
    options: &EncodeOptions,
    experiment: Arc<Experiment>,
    name: &str,
) -> Result<Value> {
    let mut options = options.clone();
    options
        .modular
        .palette_policy
        .as_mut()
        .unwrap()
        .counterfactual = Some(Arc::clone(&experiment));
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
    let mut report = json!({
        "name":name, "seconds":seconds, "events":events,
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
    println!(
        "{}",
        json!({
            "schema":1, "width":image.width(), "height":image.height(), "bits":image.bits(),
            "effort":9,"threads":args.threads,"baseline_mode":args.baseline_mode.name(),
            "target":args.target.map(target_json),"configured":configured,"trials":trials,
            "search":false,"patches_cleared":true,
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
}
