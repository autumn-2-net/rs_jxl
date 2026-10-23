//! CLI adapters for the directly ported pixel-lossless tools.
use clap::{Args, ValueEnum};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::tools::lz77::Lz77Mode;
use rs_jxl::{EncodeOptions, ModularOptions};

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
    #[arg(long,value_parser=clap::value_parser!(u32).range(1..=70000))]
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
    #[arg(long, default_value_t = 1024)]
    tree_samples: usize,
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
