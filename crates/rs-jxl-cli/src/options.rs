//! CLI adapters for the directly ported pixel-lossless tools.
use clap::{Args, ValueEnum};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::tools::lz77::Lz77Mode;
use rs_jxl::{EncodeOptions, ModularOptions};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MaBudgetMode {
    Strict,
    Bounded,
}

/// Resource controls shared by manual and source encoding, outside manual_tools.
#[derive(Debug, Args)]
#[group(id = "ma_budget", multiple = true)]
pub struct MaBudgetOptions {
    /// MA leaf exhaustion: strict errors; bounded stops optional splits, still lossless
    #[arg(long, value_enum)]
    ma_budget_mode: Option<MaBudgetMode>,
    /// Additional learned MA tree leaf cap (1..=2097152), including source/pilot
    #[arg(long, value_name = "N", value_parser = parse_ma_max_leaves)]
    ma_max_leaves: Option<usize>,
}

fn parse_ma_max_leaves(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|limit| (1..=rs_jxl::modular::ma_train::MAX_TREE_LEAVES).contains(limit))
        .ok_or_else(|| "MA leaf cap must be an integer in 1..=2097152".into())
}

impl MaBudgetOptions {
    pub fn apply(&self, options: &mut ModularOptions) {
        if let Some(mode) = self.ma_budget_mode {
            options.ma_budget_mode = Some(match mode {
                MaBudgetMode::Strict => rs_jxl::modular::ma_train::MaBudgetMode::Strict,
                MaBudgetMode::Bounded => rs_jxl::modular::ma_train::MaBudgetMode::Bounded,
            });
        }
        if let Some(limit) = self.ma_max_leaves {
            options.ma_max_leaves = Some(limit);
        }
    }
}

/// Exact MA search proofs, independent of effort and leaf-budget controls.
#[derive(Debug, Args)]
#[group(id = "ma_pruning", multiple = true)]
pub struct MaPruningOptions {
    /// Experimental exact MA partition reuse and small-node dominance pruning
    #[arg(long)]
    ma_prune: bool,
}

impl MaPruningOptions {
    pub fn apply(&self, options: &mut ModularOptions) {
        if self.ma_prune {
            options.source_property_dedup = true;
            options.source_ma_dominance = true;
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Entropy {
    Auto,
    Prefix,
    Ans,
    Source,
}
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Lz77 {
    Off,
    Rle,
    Greedy,
    Lazy,
    Optimal,
}
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Axis {
    Horizontal,
    Vertical,
}

#[derive(Debug, Args)]
#[group(id = "manual_tools", multiple = true)]
pub struct Tools {
    /// Reversible color transform 0..=41 (0 is identity)
    #[arg(long,default_value_t=6,value_parser=clap::value_parser!(u32).range(0..=41))]
    rct: u32,
    /// Predictor ID 0..=13, 14 Best (WP/gradient), 15 Variable (all)
    #[arg(long,default_value_t=14,value_parser=clap::value_parser!(u32).range(0..=15))]
    predictor: u32,
    /// Weighted predictor parameter preset 0..=4
    #[arg(long,default_value_t=0,value_parser=clap::value_parser!(u8).range(0..=4))]
    weighted_preset: u8,
    /// Automatically score the first N source reversible color transforms
    #[arg(long,value_parser=clap::value_parser!(u8).range(1..=19))]
    rct_candidates: Option<u8>,
    /// Automatically score N source weighted-predictor presets
    #[arg(long,value_parser=clap::value_parser!(u8).range(1..=5))]
    weighted_candidates: Option<u8>,
    /// Exact palette color budget; omitted disables palettes
    #[arg(long,value_parser=clap::value_parser!(u32).range(1..=70911))]
    palette_colors: Option<u32>,
    /// Stateless delta-palette predictor ID; requires --palette-colors
    #[arg(long,requires="palette_colors",value_parser=clap::value_parser!(u32).range(0..=13))]
    palette_predictor: Option<u32>,
    /// Squeeze passes, e.g. horizontal,vertical
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1)]
    squeeze: Vec<Axis>,
    #[arg(long, value_enum, default_value = "auto")]
    entropy: Entropy,
    #[arg(long, value_enum, default_value = "lazy")]
    lz77: Lz77,
    /// Manual tree-training sample target (any positive usize)
    #[arg(long, default_value_t = 1024)]
    tree_samples: usize,
    /// Manual learned tree leaf allowance (1..=2097152, the format bound)
    #[arg(long, default_value_t = 16)]
    tree_leaves: usize,
    #[arg(long, default_value_t = 48.0)]
    tree_split_threshold: f64,
    #[arg(long, default_value_t = 0)]
    previous_channels: usize,
    /// 0..=3 gives 128/256/512/1024 pixel groups
    #[arg(long,default_value_t=1,value_parser=clap::value_parser!(u8).range(0..=3))]
    group_size_shift: u8,
    #[arg(long)]
    shared_tree: bool,
    /// Encode repeated component rectangles using a lossless reference atlas
    #[arg(long)]
    patches: bool,
    #[arg(long, requires = "palette_colors")]
    global_palette: bool,
    /// Source whole-frame reversible squeeze and DC/AC partitioning
    #[arg(long)]
    responsive: bool,
    /// Three spatial passes (does not implicitly enable squeeze)
    #[arg(long)]
    progressive: bool,
    /// Source background-relative Add patches with signed working samples
    #[arg(long, conflicts_with = "patches")]
    source_patches: bool,
}

impl Tools {
    pub fn apply(&self, options: &mut EncodeOptions) -> rs_jxl::Result<()> {
        options.group_size_shift = self.group_size_shift;
        options.shared_tree = self.shared_tree;
        options.patches = self.patches.then(Default::default);
        options.global_palette = self.global_palette;
        options.responsive = self.responsive;
        options.passes = if self.progressive {
            rs_jxl::modular::layout::Passes::progressive()
        } else {
            Default::default()
        };
        options.source_patches = self.source_patches.then(Default::default);
        options.modular = ModularOptions {
            rct: Some(self.rct),
            predictor: Predictor::try_from(self.predictor)
                .map_err(|e| rs_jxl::Error::InvalidInput(e.to_string()))?,
            weighted_preset: self.weighted_preset,
            palette_colors: self.palette_colors,
            palette_predictor: self
                .palette_predictor
                .map(Predictor::try_from)
                .transpose()
                .map_err(|e| rs_jxl::Error::InvalidInput(e.to_string()))?,
            squeeze: self
                .squeeze
                .iter()
                .map(|a| matches!(a, Axis::Horizontal))
                .collect(),
            entropy: match self.entropy {
                Entropy::Auto => Coding::Auto,
                Entropy::Prefix => Coding::Prefix,
                Entropy::Ans => Coding::Ans,
                Entropy::Source => Coding::Source,
            },
            lz77: !matches!(self.lz77, Lz77::Off),
            lz77_mode: match self.lz77 {
                Lz77::Off | Lz77::Lazy => Lz77Mode::Lazy,
                Lz77::Rle => Lz77Mode::Rle,
                Lz77::Greedy => Lz77Mode::Greedy,
                Lz77::Optimal => Lz77Mode::Optimal,
            },
            tree_samples: self.tree_samples,
            max_tree_leaves: self.tree_leaves,
            split_threshold: self.tree_split_threshold,
            previous_channels: self.previous_channels,
            rct_candidates: self.rct_candidates.map(usize::from),
            weighted_candidates: self.weighted_candidates.map(usize::from),
            ..Default::default()
        };
        options.modular.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        tools: Tools,
    }

    #[test]
    fn manual_tree_options_accept_large_targets_without_extra_caps() {
        for samples in [1_048_577usize, usize::MAX] {
            for leaves in [16_385, 131_073, rs_jxl::modular::ma_train::MAX_TREE_LEAVES] {
                let args = TestCli::try_parse_from([
                    "tools",
                    "--tree-samples",
                    &samples.to_string(),
                    "--tree-leaves",
                    &leaves.to_string(),
                ])
                .unwrap();
                let mut options = EncodeOptions::default();
                args.tools.apply(&mut options).unwrap();
                assert_eq!(options.modular.tree_samples, samples);
                assert_eq!(options.modular.max_tree_leaves, leaves);
                assert_eq!(options.modular.sample_limit, None);
            }
        }
    }

    #[test]
    fn manual_tree_options_keep_nonzero_and_format_checks() {
        for (flag, value) in [
            ("--tree-samples", "0"),
            ("--tree-leaves", "0"),
            ("--tree-leaves", "2097153"),
        ] {
            let args = TestCli::try_parse_from(["tools", flag, value]).unwrap();
            assert!(args.tools.apply(&mut EncodeOptions::default()).is_err());
        }
    }

    #[test]
    fn manual_palette_options_use_the_complete_wire_range() {
        for limit in ["70001", "70911"] {
            let args = TestCli::try_parse_from(["tools", "--palette-colors", limit]).unwrap();
            args.tools.apply(&mut EncodeOptions::default()).unwrap();
        }
        for limit in ["0", "70912"] {
            assert!(TestCli::try_parse_from(["tools", "--palette-colors", limit]).is_err());
        }
    }
}
