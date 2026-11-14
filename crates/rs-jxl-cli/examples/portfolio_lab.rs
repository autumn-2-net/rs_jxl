//! Opt-in native portfolio benchmark. No corpus traversal or automatic launch.
//! Candidate probe_seconds sum elapsed probe work, which can overlap across
//! concurrent roots; scout_seconds measures wall time, not that sum.
use clap::{Parser, ValueEnum};
use rs_jxl::modular::{
    ma_train::{DescriptionCost, TreeLearning, TreeLearningReceipt},
    refinement,
};
use rs_jxl::scheduler::portfolio::{self, ALL, Candidate, Options};
use rs_jxl::scheduler::{CandidateParams, params::Override, source_tools};
use std::{fs, io::Write, path::PathBuf, sync::Arc, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DirectSourceTools {
    Baseline,
    Patch,
    Sparse,
    Prev1,
    Screened,
    SparseOnly,
    Prev1Only,
    ScreenedOnly,
}

impl DirectSourceTools {
    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Patch => "patch",
            Self::Sparse => "sparse",
            Self::Prev1 => "prev1",
            Self::Screened => "screened",
            Self::SparseOnly => "sparse-only",
            Self::Prev1Only => "prev1-only",
            Self::ScreenedOnly => "screened-only",
        }
    }

    fn build(
        self,
        image: &rs_jxl::Image,
        base: &rs_jxl::EncodeOptions,
        effort: u8,
        recipe: Candidate,
    ) -> Result<rs_jxl::EncodeOptions> {
        if !matches!(recipe, Candidate::Baseline | Candidate::Source) {
            return Err("source tool ablation requires baseline or source recipe".into());
        }
        let mut base = base.clone();
        base.source_patch_parallel_gather = matches!(
            self,
            Self::Patch | Self::Sparse | Self::Prev1 | Self::Screened
        );
        let mut params = CandidateParams::default().backend(effort)?;
        if recipe == Candidate::Source {
            params.patches = Override::On;
        }
        Ok(source_tools::Options {
            sparse_lz77: matches!(
                self,
                Self::Sparse | Self::Prev1 | Self::Screened | Self::SparseOnly
            ),
            previous_channel: matches!(self, Self::Prev1 | Self::Screened | Self::Prev1Only),
            screened_runs: matches!(self, Self::Screened | Self::ScreenedOnly),
        }
        .build(&params, image, &base)?)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DirectTreeLearning {
    Legacy,
    Cost,
    Symmetric,
}

impl From<DirectTreeLearning> for TreeLearning {
    fn from(value: DirectTreeLearning) -> Self {
        match value {
            DirectTreeLearning::Legacy => Self::Legacy,
            DirectTreeLearning::Cost => Self::Cost,
            DirectTreeLearning::Symmetric => Self::Symmetric,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DirectMaProfile {
    Legacy,
    Half,
    Checked,
    HalfChecked,
    SinglePass,
    HalfSinglePass,
}

impl DirectMaProfile {
    fn configure(self, options: &mut rs_jxl::EncodeOptions) {
        options.modular.refinement.offsets = match self {
            Self::Legacy | Self::Half => None,
            Self::Checked | Self::HalfChecked => Some(refinement::OffsetOptions {
                mode: refinement::OffsetMode::Distribution,
                ..Default::default()
            }),
            Self::SinglePass | Self::HalfSinglePass => Some(refinement::OffsetOptions {
                mode: refinement::OffsetMode::DistributionSinglePass,
                ..Default::default()
            }),
        };
        options.modular.source_ma_scoring.description = match self {
            Self::Legacy | Self::Checked | Self::SinglePass => DescriptionCost::Disabled,
            Self::Half | Self::HalfChecked | Self::HalfSinglePass => DescriptionCost::Half,
        };
        if let Some(reference) = &mut options.source_patch_reference {
            reference.modular.refinement.offsets = options.modular.refinement.offsets.clone();
            reference.modular.source_ma_scoring.description =
                options.modular.source_ma_scoring.description;
        }
    }
}

#[derive(Parser)]
#[command(
    after_help = "Candidate probe_seconds sum elapsed probe work and can exceed scout_seconds when roots run concurrently; scout_seconds is wall time."
)]
struct Args {
    input: Option<PathBuf>,
    #[arg(long, required_unless_present_any = ["plan_only", "list_candidates"])]
    output: Option<PathBuf>,
    /// Direct recipes support e1..e11; portfolio search remains e8..e10.
    #[arg(long, default_value_t = 9, value_parser = clap::value_parser!(u8).range(1..=11))]
    effort: u8,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    /// Run independent probe roots serially instead of concurrently.
    #[arg(long)]
    serial_probes: bool,
    /// Preparation cache budget in MiB; zero disables reuse.
    #[arg(long, default_value_t = 128)]
    preparation_cache_mib: usize,
    /// Disable safe output-bound exits for losing full finalists.
    #[arg(long)]
    no_finalist_bound: bool,
    #[arg(long, default_value_t = 48)]
    max_probe_encodes: usize,
    #[arg(long, default_value_t = 2)]
    max_finalists: usize,
    #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u16).range(32..=512))]
    probe_side: u16,
    #[arg(long, default_value_t = 0.01)]
    uncertainty: f64,
    /// Explicit comma-separated subset; baseline is always retained.
    #[arg(long, value_delimiter = ',')]
    candidates: Vec<String>,
    /// Encode one recipe once, without scout, fallback, bound or preparation cache.
    #[arg(long, conflicts_with_all = ["plan_only", "list_candidates", "candidates",
        "serial_probes", "preparation_cache_mib", "no_finalist_bound",
        "max_probe_encodes", "max_finalists", "probe_side", "uncertainty"])]
    direct_candidate: Option<String>,
    /// Override MA defaults for a direct benchmark only, without changing recipes.
    #[arg(long, value_enum, requires = "direct_candidate", conflicts_with_all = ["plan_only", "list_candidates"])]
    direct_ma_profile: Option<DirectMaProfile>,
    /// Override tree learning independently of the direct MA profile.
    #[arg(long, value_enum, requires = "direct_candidate", conflicts_with_all = ["plan_only", "list_candidates"])]
    direct_tree_learning: Option<DirectTreeLearning>,
    /// Cumulative/isolated e9 tools; one encode, never a best-of tournament.
    #[arg(long, value_enum, requires = "direct_candidate",
        conflicts_with_all = ["plan_only", "list_candidates", "direct_ma_profile", "direct_tree_learning"])]
    direct_source_tools: Option<DirectSourceTools>,
    #[arg(long, conflicts_with = "output")]
    plan_only: bool,
    #[arg(long, conflicts_with_all = ["output", "input"])]
    list_candidates: bool,
}

fn candidate(name: &str) -> Result<Candidate> {
    ALL.iter()
        .copied()
        .find(|c| c.name() == name)
        .ok_or_else(|| format!("unknown candidate {name}; use --list-candidates").into())
}

fn encode_direct(args: &Args, image: &rs_jxl::Image, recipe: Candidate) -> Result<()> {
    let mut base = rs_jxl::EncodeOptions {
        threads: usize::from(args.threads),
        ..Default::default()
    };
    if args.direct_ma_profile.is_some() {
        base.modular.source_ma_defaults = false;
    }
    let start = Instant::now();
    // Ordinary e1 must use the same FJXL eligibility/fallback dispatch as the
    // public native API. Explicit experiments continue to target Modular.
    let native = if args.effort == 1
        && recipe == Candidate::Baseline
        && args.direct_source_tools.is_none()
        && args.direct_ma_profile.is_none()
        && args.direct_tree_learning.is_none()
    {
        Some(rs_jxl::scheduler::native::encode_native_candidate(
            &rs_jxl::scheduler::CandidateParams::default().backend(1)?,
            image,
            &base,
            Default::default(),
        )?)
    } else {
        None
    };
    let (mut configured, encoded, implementation) = if let Some(output) = native {
        (
            output.artifact.options,
            Some(output.bytes),
            output.artifact.implementation,
        )
    } else {
        let configured = if let Some(profile) = args.direct_source_tools {
            profile.build(image, &base, args.effort, recipe)?
        } else {
            recipe.build(image, &base, args.effort)?
        };
        (
            configured,
            None,
            rs_jxl::scheduler::NativeImplementation::Modular,
        )
    };
    if let Some(profile) = args.direct_ma_profile {
        profile.configure(&mut configured);
    }
    if let Some(mode) = args.direct_tree_learning {
        configured.modular.tree_learning = mode.into();
        if let Some(reference) = &mut configured.source_patch_reference {
            reference.modular.tree_learning = mode.into();
        }
    }
    let receipt = Arc::new(refinement::Receipt::default());
    let learning_receipt = Arc::new(TreeLearningReceipt::default());
    configured.modular.refinement.receipt = Some(Arc::clone(&receipt));
    configured.modular.tree_learning_receipt = Some(Arc::clone(&learning_receipt));
    if let Some(reference) = &mut configured.source_patch_reference {
        reference.modular.refinement.receipt = Some(Arc::clone(&receipt));
        reference.modular.tree_learning_receipt = Some(Arc::clone(&learning_receipt));
    }
    let bytes = match encoded {
        Some(bytes) => bytes,
        None => rs_jxl::encode(image, &configured)?,
    };
    let seconds = start.elapsed().as_secs_f64();
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("invalid encode duration".into());
    }
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.output.as_ref().ok_or("output is required")?)?
        .write_all(&bytes)?;
    let pixels = f64::from(image.width()) * f64::from(image.height());
    let work = receipt.snapshot();
    let learning = learning_receipt.snapshot();
    let learning_mode = match configured.modular.tree_learning {
        TreeLearning::Legacy => "legacy",
        TreeLearning::Cost => "cost",
        TreeLearning::Symmetric => "symmetric",
    };
    let offset_mode = configured
        .modular
        .refinement
        .offsets
        .as_ref()
        .map(|o| match o.mode {
            refinement::OffsetMode::Mean => "mean",
            refinement::OffsetMode::Median => "median",
            refinement::OffsetMode::Entropy => "entropy",
            refinement::OffsetMode::Distribution => "distribution",
            refinement::OffsetMode::DistributionSinglePass => "distribution-single-pass",
        });
    let actual_profile = match (
        configured.modular.source_ma_scoring.description,
        offset_mode,
    ) {
        (DescriptionCost::Disabled, None) => "legacy",
        (DescriptionCost::Half, None) => "half",
        (DescriptionCost::Disabled, Some("distribution")) => "checked",
        (DescriptionCost::Half, Some("distribution")) => "half-checked",
        (DescriptionCost::Disabled, Some("distribution-single-pass")) => "single-pass",
        (DescriptionCost::Half, Some("distribution-single-pass")) => "half-single-pass",
        _ => "custom",
    };
    let refinement = serde_json::json!({
        "offset_mode": offset_mode,
        "offset_sample_limit": configured.modular.refinement.offsets.as_ref().map(|o| o.samples_per_leaf),
        "domains": work.domains, "models": work.models,
        "offset_single_pass_domains": work.offset_single_pass_domains,
        "offset_applied": work.offset_applied,
        "offset_proposals": work.offset_proposals, "offset_wins": work.offset_wins,
        "offset_leaves": work.offset_leaves, "offset_samples": work.offset_samples,
        "offset_population": work.offset_population, "offset_candidates": work.offset_candidates,
        "offset_token_fits": work.offset_token_fits,
        "offset_reused_tokens": work.offset_reused_tokens,
        "offset_raw_reused_tokens": work.offset_raw_reused_tokens,
        "saved_bits": work.saved_bits,
        "offset_seconds": work.offset_seconds, "refit_seconds": work.refit_seconds,
        "model_seconds": work.model_seconds, "write_seconds": work.write_seconds,
    });
    println!(
        "{}",
        serde_json::json!({
            "schema": 1, "direct": true, "candidate": recipe.name(),
            "implementation": format!("{implementation:?}"),
            "effort": args.effort, "threads": args.threads,
            "width": image.width(), "height": image.height(), "bits": image.bits(),
            "bytes": bytes.len(), "bpp": bytes.len() as f64 * 8.0 / pixels,
            "seconds": seconds, "mpix_per_second": pixels / 1e6 / seconds,
            "encode_calls": 1, "preparation_cache_bytes": 0,
            "direct_ma_profile": actual_profile,
            "direct_tree_learning": learning_mode,
            "tree_learning": {
                "learners": learning.learners, "records": learning.records,
                "cost_splits": learning.cost_splits,
                "symmetric_splits": learning.symmetric_splits,
            },
            "source_ma_defaults": configured.modular.source_ma_defaults,
            "source_tools": {
                "profile": args.direct_source_tools.map(DirectSourceTools::name),
                "patch_parallel_gather": configured.source_patch_parallel_gather,
                "previous_channels": configured.modular.previous_channels,
                "max_previous_channels": args.direct_source_tools.map(|p| u8::from(matches!(p,
                    DirectSourceTools::Prev1 | DirectSourceTools::Screened | DirectSourceTools::Prev1Only))),
                "lz_quick": configured.modular.source_lz77.as_ref().map(|c| format!("{:?}", c.quick)),
                "palette_order": configured.modular.palette_policy.as_ref().map(|p| format!("{:?}", p.tools.order)),
            },
            "description_cost": format!("{:?}", configured.modular.source_ma_scoring.description),
            "refinement": refinement,
            "timing": "recipe projection and one complete native encode, including pool setup; excludes image IO",
        })
    );
    Ok(())
}

fn run(args: Args) -> Result<()> {
    if args.list_candidates {
        println!(
            "{}",
            serde_json::json!(
                ALL.iter()
                    .map(|c| serde_json::json!({
                        "name": c.name(), "family": format!("{:?}", c.family()),
                    }))
                    .collect::<Vec<_>>()
            )
        );
        return Ok(());
    }
    let direct = args
        .direct_candidate
        .as_deref()
        .map(candidate)
        .transpose()?;
    if direct.is_none() && !(8..=10).contains(&args.effort) {
        return Err(
            "portfolio search effort must be 8..=10; use --direct-candidate for other efforts"
                .into(),
        );
    }
    let input = args.input.as_ref().ok_or("input is required")?;
    if args.output.as_ref().is_some_and(|p| p.exists()) {
        return Err("output already exists".into());
    }
    let image = rs_jxl_image::decode(&fs::read(input)?, true)?.image;
    if let Some(recipe) = direct {
        return encode_direct(&args, &image, recipe);
    }
    let options = Options {
        effort: args.effort,
        threads: usize::from(args.threads),
        parallel_probes: !args.serial_probes,
        preparation_cache_bytes: args
            .preparation_cache_mib
            .checked_mul(1 << 20)
            .ok_or("preparation cache MiB exceeds addressable bytes")?,
        bound_finalists: !args.no_finalist_bound,
        max_probe_encodes: args.max_probe_encodes,
        max_finalists: args.max_finalists,
        probe_side: usize::from(args.probe_side),
        uncertainty: args.uncertainty,
        candidates: if args.candidates.is_empty() {
            ALL.to_vec()
        } else {
            args.candidates
                .iter()
                .map(|s| candidate(s))
                .collect::<Result<_>>()?
        },
        ..Default::default()
    };
    let (bytes, report) = if args.plan_only {
        (None, portfolio::plan(&image, &options)?)
    } else {
        let result = portfolio::encode(&image, &options)?;
        (Some(result.bytes), result.report)
    };
    if let Some(bytes) = &bytes {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.output.ok_or("output is required")?)?
            .write_all(bytes)?;
    }
    let pixels = f64::from(image.width()) * f64::from(image.height());
    let baseline = &report.candidates[0];
    let f = &report.features;
    println!(
        "{}",
        serde_json::json!({
            "schema": 1, "plan_only": args.plan_only,
            "effort": args.effort, "threads": args.threads,
            "parallel_probes": options.parallel_probes,
            "preparation_cache_bytes": options.preparation_cache_bytes,
            "bound_finalists": options.bound_finalists,
            "width": image.width(), "height": image.height(), "bits": image.bits(),
            "bytes": bytes.as_ref().map(Vec::len),
            "bpp": bytes.as_ref().map(|b| b.len() as f64 * 8.0 / pixels),
            "seconds": report.seconds,
            "mpix_per_second": (report.seconds > 0.0).then(|| pixels / 1e6 / report.seconds),
            "baseline": {"bytes": baseline.full_bytes, "seconds": baseline.full_seconds,
                "bpp": baseline.full_bytes.map(|n| n as f64 * 8.0 / pixels)},
            "scout_seconds": report.scout_seconds,
            "probe_workers": report.probe_workers,
            "preparation_reuse": {
                "hits": report.preparation_reuse.hits,
                "misses": report.preparation_reuse.misses,
                "duplicate_builds": report.preparation_reuse.duplicate_builds,
                "bypasses": report.preparation_reuse.bypasses,
                "entries": report.preparation_reuse.entries,
                "retained_bytes": report.preparation_reuse.retained_bytes,
                "peak_bytes": report.preparation_reuse.peak_bytes,
            },
            "probe_encodes": report.probe_encodes, "probe_pixels": report.probe_pixels,
            "full_encodes": report.full_encodes, "winner": report.winner.name(),
            "features": {"samples": f.samples, "unique_fraction": f.unique_fraction,
                "flat_fraction": f.flat_fraction, "gradient_zero_fraction": f.gradient_zero_fraction,
                "residual_bias": f.residual_bias, "cross_channel_gain": f.cross_channel_gain,
                "residual_unique_fraction": f.residual_unique_fraction,
            "residual_zero_fraction": f.residual_zero_fraction,
                "repeated_tile_fraction": f.repeated_tile_fraction, "sampled_tiles": f.sampled_tiles},
            "regions": report.regions.iter().map(|r| serde_json::json!({
                "x": r.x,"y": r.y,"width": r.width,"height": r.height,
            })).collect::<Vec<_>>(),
            "candidates": report.candidates.iter().map(|c| serde_json::json!({
                "name": c.candidate.name(), "family": format!("{:?}", c.candidate.family()),
                "decision": c.decision, "reason": c.reason, "priority": c.priority,
                "probe_bytes": c.probe_bytes, "probe_seconds": c.probe_seconds,
                "full_bytes": c.full_bytes, "full_seconds": c.full_seconds, "error": c.error,
                "full_lower_bound": c.full_lower_bound,
            })).collect::<Vec<_>>(),
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    let result = run(Args::parse());
    if let Some(profile) = rs_jxl::profile::report() {
        eprintln!("{profile}");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_baseline_accepts_all_source_efforts_without_relaxing_search() {
        let image = rs_jxl::Image::new(
            16,
            16,
            rs_jxl::PixelLayout::Rgb,
            rs_jxl::PixelData::U8(vec![1; 16 * 16 * 3]),
        )
        .unwrap();
        for effort in 1..=11 {
            let level = effort.to_string();
            let args = Args::try_parse_from([
                "lab",
                "image.png",
                "--output",
                "out.jxl",
                "--effort",
                &level,
                "--direct-candidate",
                "baseline",
            ])
            .unwrap();
            assert_eq!(args.effort, effort);
            Candidate::Baseline
                .build(&image, &rs_jxl::EncodeOptions::default(), effort)
                .unwrap();
            if !(8..=10).contains(&effort) {
                let search =
                    Args::try_parse_from(["lab", "image.png", "--effort", &level, "--plan-only"])
                        .unwrap();
                assert!(
                    run(search)
                        .unwrap_err()
                        .to_string()
                        .contains("portfolio search effort")
                );
            }
        }
    }

    #[test]
    fn direct_tools_require_an_explicit_direct_e9_route() {
        let image = rs_jxl::Image::new(
            16,
            16,
            rs_jxl::PixelLayout::Rgb,
            rs_jxl::PixelData::U8(vec![1; 16 * 16 * 3]),
        )
        .unwrap();
        for profile in DirectSourceTools::value_variants() {
            let words = [
                "lab",
                "image.png",
                "--output",
                "out.jxl",
                "--direct-source-tools",
                profile.name(),
            ];
            assert!(Args::try_parse_from(words).is_err());
            let args =
                Args::try_parse_from(words.into_iter().chain(["--direct-candidate", "baseline"]))
                    .unwrap();
            assert_eq!(args.direct_source_tools, Some(*profile));
            let options = profile
                .build(
                    &image,
                    &rs_jxl::EncodeOptions::default(),
                    9,
                    Candidate::Baseline,
                )
                .unwrap();
            assert_eq!(
                options.source_patch_parallel_gather,
                matches!(
                    profile,
                    DirectSourceTools::Patch
                        | DirectSourceTools::Sparse
                        | DirectSourceTools::Prev1
                        | DirectSourceTools::Screened
                )
            );
            assert!(
                profile
                    .build(
                        &image,
                        &rs_jxl::EncodeOptions::default(),
                        8,
                        Candidate::Baseline
                    )
                    .is_err()
            );
            assert!(
                profile
                    .build(
                        &image,
                        &rs_jxl::EncodeOptions::default(),
                        9,
                        Candidate::Prev1
                    )
                    .is_err()
            );
        }
    }
    #[test]
    fn parsing_requires_explicit_output_or_analysis() {
        assert!(Args::try_parse_from(["lab", "input.png"]).is_err());
        assert!(Args::try_parse_from(["lab", "--list-candidates"]).is_ok());
        assert!(Args::try_parse_from(["lab", "input.png", "--plan-only"]).is_ok());
        assert!(
            Args::try_parse_from(["lab", "input.png", "--plan-only", "--output", "out.jxl"])
                .is_err()
        );
        assert!(candidate("made-up-tool").is_err());
    }

    #[test]
    fn direct_recipe_requires_output_and_rejects_ignored_scout_controls() {
        let base = ["lab", "input.png", "--direct-candidate", "baseline"];
        assert!(Args::try_parse_from(base).is_err());
        let args = Args::try_parse_from(base.into_iter().chain(["--output", "out.jxl"])).unwrap();
        assert_eq!(args.direct_candidate.as_deref(), Some("baseline"));
        assert!(args.direct_ma_profile.is_none());
        for extra in [
            vec!["--plan-only"],
            vec!["--list-candidates"],
            vec!["--candidates", "prev2"],
            vec!["--max-finalists", "2"],
            vec!["--max-probe-encodes", "48"],
            vec!["--probe-side", "128"],
            vec!["--uncertainty", "0.01"],
            vec!["--preparation-cache-mib", "0"],
            vec!["--serial-probes"],
            vec!["--no-finalist-bound"],
        ] {
            assert!(
                Args::try_parse_from(base.into_iter().chain(["--output", "out.jxl"]).chain(extra))
                    .is_err()
            );
        }
    }

    #[test]
    fn direct_ma_profiles_require_direct_candidate_and_accept_only_known_names() {
        for (name, expected) in [
            ("legacy", DirectMaProfile::Legacy),
            ("half", DirectMaProfile::Half),
            ("checked", DirectMaProfile::Checked),
            ("half-checked", DirectMaProfile::HalfChecked),
            ("single-pass", DirectMaProfile::SinglePass),
            ("half-single-pass", DirectMaProfile::HalfSinglePass),
        ] {
            let base = ["lab", "input.png", "--direct-ma-profile", name];
            assert!(Args::try_parse_from(base.into_iter().chain(["--output", "out.jxl"])).is_err());
            assert!(Args::try_parse_from(base.into_iter().chain(["--plan-only"])).is_err());
            let args = Args::try_parse_from(base.into_iter().chain([
                "--direct-candidate",
                "baseline",
                "--output",
                "out.jxl",
            ]))
            .unwrap();
            assert_eq!(args.direct_ma_profile, Some(expected));
        }
        assert!(
            Args::try_parse_from([
                "lab",
                "input.png",
                "--direct-candidate",
                "baseline",
                "--output",
                "out.jxl",
                "--direct-ma-profile",
                "unknown",
            ])
            .is_err()
        );
    }

    #[test]
    fn explicit_ma_profiles_replace_recipe_offset_and_description_settings() {
        for (profile, description, offsets) in [
            (DirectMaProfile::Legacy, DescriptionCost::Disabled, None),
            (DirectMaProfile::Half, DescriptionCost::Half, None),
            (
                DirectMaProfile::Checked,
                DescriptionCost::Disabled,
                Some(refinement::OffsetMode::Distribution),
            ),
            (
                DirectMaProfile::HalfChecked,
                DescriptionCost::Half,
                Some(refinement::OffsetMode::Distribution),
            ),
            (
                DirectMaProfile::SinglePass,
                DescriptionCost::Disabled,
                Some(refinement::OffsetMode::DistributionSinglePass),
            ),
            (
                DirectMaProfile::HalfSinglePass,
                DescriptionCost::Half,
                Some(refinement::OffsetMode::DistributionSinglePass),
            ),
        ] {
            let mut options = rs_jxl::EncodeOptions::default();
            options.modular.source_ma_scoring.description = DescriptionCost::Full;
            options.modular.refinement.offsets = Some(refinement::OffsetOptions::default());
            profile.configure(&mut options);
            assert_eq!(options.modular.source_ma_scoring.description, description);
            assert_eq!(
                options.modular.refinement.offsets.as_ref().map(|o| o.mode),
                offsets
            );
        }
    }

    #[test]
    fn direct_tree_learning_requires_direct_and_combines_with_ma_profiles() {
        for (name, expected) in [
            ("legacy", DirectTreeLearning::Legacy),
            ("cost", DirectTreeLearning::Cost),
            ("symmetric", DirectTreeLearning::Symmetric),
        ] {
            let base = ["lab", "input.png", "--direct-tree-learning", name];
            assert!(Args::try_parse_from(base.into_iter().chain(["--output", "out.jxl"])).is_err());
            assert!(Args::try_parse_from(base.into_iter().chain(["--plan-only"])).is_err());
            for profile in [
                "legacy",
                "half",
                "checked",
                "half-checked",
                "single-pass",
                "half-single-pass",
            ] {
                let args = Args::try_parse_from(base.into_iter().chain([
                    "--direct-candidate",
                    "baseline",
                    "--output",
                    "out.jxl",
                    "--direct-ma-profile",
                    profile,
                ]))
                .unwrap();
                assert_eq!(args.direct_tree_learning, Some(expected));
            }
        }
        assert!(
            Args::try_parse_from([
                "lab",
                "input.png",
                "--direct-candidate",
                "baseline",
                "--output",
                "out.jxl",
                "--direct-tree-learning",
                "unknown",
            ])
            .is_err()
        );
    }

    #[test]
    fn performance_controls_default_on_and_allow_disabling() {
        let args = Args::try_parse_from(["lab", "input.png", "--plan-only"]).unwrap();
        assert!(!args.serial_probes);
        assert_eq!(args.preparation_cache_mib, 128);
        assert!(!args.no_finalist_bound);
        let args = Args::try_parse_from([
            "lab",
            "input.png",
            "--plan-only",
            "--serial-probes",
            "--preparation-cache-mib",
            "0",
            "--no-finalist-bound",
        ])
        .unwrap();
        assert!(args.serial_probes);
        assert_eq!(args.preparation_cache_mib, 0);
        assert!(args.no_finalist_bound);
        assert!(
            Args::try_parse_from([
                "lab",
                "input.png",
                "--plan-only",
                "--preparation-cache-mib",
                "-1",
            ])
            .is_err()
        );
    }
}
