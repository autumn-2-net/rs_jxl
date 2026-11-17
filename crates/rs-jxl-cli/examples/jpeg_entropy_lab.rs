//! Isolated JPEG compression experiments; original JPEG bytes are the oracle.
use std::{fs, io::Write, path::PathBuf, sync::Arc, time::Instant};

use clap::Parser;
use rs_jxl::jpeg::{self, TranscodeOptions};

#[derive(Parser)]
struct Args {
    input: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 9, value_parser = clap::value_parser!(u8).range(1..=11))]
    effort: u8,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    /// Expanded standard AC HybridUint configurations, exact container fallback.
    #[arg(long)]
    head_tail: bool,
    /// Choose a representable actual CfL maximum; one-pass, no size fallback.
    #[arg(long, conflicts_with = "head_tail")]
    cfl_peak: bool,
    /// Lend the existing pool to auxiliary MA split search.
    #[arg(long)]
    cooperative: bool,
    /// Native symmetric MA with half description cost, without a second encode.
    #[arg(long, conflicts_with = "head_tail")]
    symmetric_ma: bool,
    #[arg(long, conflicts_with = "head_tail")]
    paired_luma: bool,
    #[arg(long, default_value = "source", value_parser = ["source", "off", "fast", "optimal", "fast-forced", "optimal-forced"], conflicts_with = "head_tail")]
    ac_lz: String,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=4), conflicts_with = "head_tail")]
    ac_histograms: u8,
    #[arg(long, default_value = "source", value_parser = ["source", "channel", "shared-chroma", "paired-luma", "quantile4", "quantile5", "activity4", "activity5"], conflicts_with_all = ["head_tail", "paired_luma"])]
    block_context: String,
    #[arg(long, default_value = "source", value_parser = ["source", "zigzag", "edge-first", "edge-last", "band-edge-first", "band-edge-last"], conflicts_with = "head_tail")]
    ac_order: String,
    /// Inspect selected AC symbols and extra bits; adds diagnostic runtime.
    #[arg(long, conflicts_with = "head_tail")]
    audit_ac: bool,
}

impl Args {
    fn tuning(&self) -> jpeg::tuning::Options {
        jpeg::tuning::Options {
            cfl_fit: if self.cfl_peak {
                jpeg::cfl::CflFitPolicy::ActualMaximum
            } else {
                Default::default()
            },
            symmetric_ma: self.symmetric_ma,
            paired_luma: self.paired_luma,
            ac_lz: match self.ac_lz.as_str() {
                "off" => jpeg::tuning::AcLzSearch::Off,
                "fast" => jpeg::tuning::AcLzSearch::Fast,
                "optimal" => jpeg::tuning::AcLzSearch::Optimal,
                "fast-forced" => jpeg::tuning::AcLzSearch::FastForced,
                "optimal-forced" => jpeg::tuning::AcLzSearch::OptimalForced,
                _ => jpeg::tuning::AcLzSearch::Source,
            },
            ac_histograms: usize::from(self.ac_histograms),
            block_context: match self.block_context.as_str() {
                "channel" => jpeg::tuning::BlockContextPolicy::ChannelOnly,
                "shared-chroma" => jpeg::tuning::BlockContextPolicy::SharedChroma,
                "paired-luma" => jpeg::tuning::BlockContextPolicy::PairedLuma,
                "quantile4" => jpeg::tuning::BlockContextPolicy::Quantile4,
                "quantile5" => jpeg::tuning::BlockContextPolicy::Quantile5,
                "activity4" => jpeg::tuning::BlockContextPolicy::Activity4,
                "activity5" => jpeg::tuning::BlockContextPolicy::Activity5,
                _ => Default::default(),
            },
            ac_order: match self.ac_order.as_str() {
                "zigzag" => jpeg::tuning::AcOrderPolicy::Zigzag,
                "edge-first" => jpeg::tuning::AcOrderPolicy::EdgeFirst,
                "edge-last" => jpeg::tuning::AcOrderPolicy::EdgeLast,
                "band-edge-first" => jpeg::tuning::AcOrderPolicy::BandEdgeFirst,
                "band-edge-last" => jpeg::tuning::AcOrderPolicy::BandEdgeLast,
                _ => Default::default(),
            },
        }
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let input = fs::read(&args.input)?;
    let (width, height) = {
        let info = jpeg::reader::read_jpeg(&input)?;
        (info.width, info.height)
    };
    let options = TranscodeOptions {
        effort: args.effort,
        threads: usize::from(args.threads),
        cooperative_ma: args.cooperative.then(Default::default),
        ..Default::default()
    };
    let receipt = Arc::new(jpeg::head_tail::Receipt::default());
    let start = Instant::now();
    let (output, sizes, audit) = if !args.head_tail {
        let tuning = args.tuning();
        if args.audit_ac {
            let (output, sizes, audit) = jpeg::tuning::transcode_audited(&input, &options, tuning)?;
            (output, Some(sizes), Some(audit))
        } else {
            let (output, sizes) = jpeg::tuning::transcode_measured(&input, &options, tuning)?;
            (output, Some(sizes), None)
        }
    } else {
        (
            jpeg::head_tail::transcode(&input, &options, &receipt)?,
            None,
            None,
        )
    };
    let seconds = start.elapsed().as_secs_f64();
    if let Some(report) = rs_jxl::profile::report() {
        eprint!("{report}");
    }
    let reconstructed = rs_jxl::decode::reconstruct_jpeg(&output, &Default::default())?;
    if reconstructed != input {
        return Err("native JPEG reconstruction differs from original bytes".into());
    }
    if let Some(path) = args.output {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(&output)?;
    }
    let head_tail = args.head_tail.then(|| {
        let r = receipt.snapshot();
        serde_json::json!({"attempted":r.attempted,"candidate":r.candidate,"selected":r.selected,
            "histograms":r.histograms,"changed_histograms":r.changed_histograms,
            "configurations":r.configurations,"baseline_bytes":r.baseline_bytes,
            "candidate_bytes":r.candidate_bytes,"search_seconds":r.search_seconds,"write_seconds":r.write_seconds})
    });
    let pixels = f64::from(width) * f64::from(height);
    let ac_lz_enabled = sizes.and_then(|s| s.ac_lz_enabled);
    let ac_histograms_used = sizes.and_then(|s| s.ac_histogram_groups);
    let size_breakdown = sizes.map(|s| serde_json::json!({
        "global_bits":s.global_bits,"aux_model_bits":s.aux_model_bits,"dc_bits":s.dc_bits,
        "ac_metadata_bits":s.ac_metadata_bits,"quant_bits":s.quant_bits,"ac_model_bits":s.ac_model_bits,
        "ac_data_bits":s.ac_data_bits,"padding_bits":s.padding_bits,"header_bits":s.header_bits,
        "reconstruction_bits":s.reconstruction_bits,"metadata_bits":s.metadata_bits,"container_bits":s.container_bits
    }));
    println!(
        "{}",
        serde_json::json!({"bytes":output.len(),"seconds":seconds,"width":width,"height":height,
        "bpp":output.len() as f64*8.0/pixels,"mpix_per_second":pixels/1e6/seconds,
        "jpeg_bytes":input.len(),"native_verified":true,"head_tail":head_tail,"cfl_peak":args.cfl_peak,
        "cooperative":args.cooperative,"symmetric_ma":args.symmetric_ma,
        "paired_luma":args.paired_luma,
        "ac_lz":args.ac_lz,"ac_lz_enabled":ac_lz_enabled,"size_breakdown":size_breakdown,
        "ac_histograms":args.ac_histograms,"ac_histograms_used":ac_histograms_used,
        "block_context":args.block_context,"ac_order":args.ac_order,
        "ac_block_contexts":sizes.and_then(|s| s.ac_block_contexts),
        "dc_thresholds":sizes.and_then(|s| s.dc_thresholds),
        "audit_ac":args.audit_ac,"ac_audit":audit.as_ref().map(audit_json),
        "timing":if args.audit_ac {
            "native JPEG parse and encode INCLUDING diagnostic audit; not normal encoding speed"
        } else {
            "native JPEG parse and encode; excludes disk I/O, dimension inspection and reconstruction"
        }})
    );
    Ok(())
}

fn audit_json(a: &jpeg::ac_audit::AcAudit) -> serde_json::Value {
    let e = &a.entropy;
    serde_json::json!({
        "streams":e.streams,"nonempty_streams":e.nonempty_streams,"tokens":e.tokens,
        "raw_contexts":e.raw_contexts,"active_contexts":e.active_contexts,
        "histograms":e.histograms,"ans":e.ans,"lz77":e.lz77,
        "extra_bits":e.extra_bits,"model_symbol_bits":e.model_symbol_bits,
        "cluster_entropy_bits":e.cluster_entropy_bits,
        "raw_context_entropy_bits":e.raw_context_entropy_bits,
        "extra_bit_entropy_bits":e.extra_bit_entropy_bits,
        "selector_bits":a.selector_bits,"seconds":a.seconds
    })
}

fn main() -> std::process::ExitCode {
    match run(Args::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[test]
fn experiment_is_explicit_and_validates_effort_and_threads() {
    let defaults = Args::try_parse_from(["lab", "in.jpg"]).unwrap();
    assert_eq!(defaults.ac_lz, "source");
    assert_eq!(defaults.ac_histograms, 1);
    assert_eq!(defaults.block_context, "source");
    assert_eq!(defaults.ac_order, "source");
    let tuning = defaults.tuning();
    assert_eq!(tuning.ac_lz, jpeg::tuning::AcLzSearch::Source);
    assert_eq!(
        tuning.block_context,
        jpeg::tuning::BlockContextPolicy::Source
    );
    assert_eq!(tuning.ac_order, jpeg::tuning::AcOrderPolicy::Source);
    assert_eq!(tuning.ac_histograms, 1);
    assert!(
        !defaults.head_tail
            && !defaults.cfl_peak
            && !defaults.symmetric_ma
            && !defaults.cooperative
            && !defaults.paired_luma
            && !defaults.audit_ac
    );
    for flag in [
        "--cfl-peak",
        "--symmetric-ma",
        "--paired-luma",
        "--ac-lz=fast",
        "--ac-histograms=2",
        "--block-context=channel",
        "--ac-order=zigzag",
        "--audit-ac",
    ] {
        assert!(Args::try_parse_from(["lab", "in.jpg", "--head-tail", flag]).is_err());
    }
    assert!(
        Args::try_parse_from(["lab", "in.jpg", "--head-tail"])
            .unwrap()
            .head_tail
    );
    for args in [
        ["--effort", "0"],
        ["--effort", "12"],
        ["--threads", "0"],
        ["--threads", "257"],
        ["--ac-histograms", "0"],
        ["--ac-histograms", "5"],
        ["--ac-lz", "unknown"],
        ["--block-context", "unknown"],
        ["--ac-order", "unknown"],
    ] {
        assert!(Args::try_parse_from(["lab", "in.jpg", args[0], args[1]]).is_err());
    }
    assert!(Args::try_parse_from(["lab", "in.jpg", "--paired-luma"]).is_ok());
    assert!(
        Args::try_parse_from(["lab", "in.jpg", "--paired-luma", "--block-context=channel"])
            .is_err()
    );
}

#[test]
fn explicit_arguments_map_to_native_policies() {
    let args = Args::try_parse_from([
        "lab",
        "in.jpg",
        "--cfl-peak",
        "--symmetric-ma",
        "--audit-ac",
        "--ac-lz=optimal-forced",
        "--ac-histograms=4",
        "--block-context=activity5",
        "--ac-order=band-edge-last",
    ])
    .unwrap();
    let tuning = args.tuning();
    assert!(matches!(
        tuning.cfl_fit,
        jpeg::cfl::CflFitPolicy::ActualMaximum
    ));
    assert!(tuning.symmetric_ma);
    assert!(!tuning.paired_luma);
    assert_eq!(tuning.ac_lz, jpeg::tuning::AcLzSearch::OptimalForced);
    assert_eq!(tuning.ac_histograms, 4);
    assert_eq!(
        tuning.block_context,
        jpeg::tuning::BlockContextPolicy::Activity5
    );
    assert_eq!(tuning.ac_order, jpeg::tuning::AcOrderPolicy::BandEdgeLast);
    assert!(args.audit_ac);
}
