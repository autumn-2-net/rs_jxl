// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Modular channel traversal and local group assembly from enc_encoding.cc,
//! context_predict.h and enc_modular.cc. Tools are selected explicitly here;
//! the complete optimized image-level scheduler remains a separate migration.
use super::group::{self, Channel, Transform};
use super::{
    ma_train,
    predictor::{Neighbors, Predictor, predict_one},
    tree::{Properties, Tree},
    weighted,
};
use crate::entropy::stream::{Coding, Model, Token};
use crate::tools::lz77::{self, Command, Lz77Options};
use crate::{Error, Image, Result, bit_writer::BitWriter};

#[path = "receipt.rs"]
pub mod receipt;
#[path = "receipt_cost.rs"]
pub mod receipt_cost;

#[cfg(test)]
#[path = "receipt_cost_tests.rs"]
mod receipt_cost_tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedTree {
    Zero,
    Weighted,
    Gradient,
    GradientLeaf,
}

#[derive(Debug, Clone)]
pub struct ModularOptions {
    pub predictor: Predictor,
    pub rct: Option<u32>,
    pub weighted_preset: u8,
    pub tree_samples: usize,
    pub max_tree_leaves: usize,
    /// Separate complete-tree ceiling for source learning. None preserves the
    /// legacy max_tree_leaves ceiling; Some is bounded by decoder node limits.
    pub max_merged_tree_leaves: Option<usize>,
    /// Optional MA exhaustion policy; None keeps source strict / legacy bounded.
    pub ma_budget_mode: Option<ma_train::MaBudgetMode>,
    /// Additional learned-tree cap, 1..=MAX_TREE_LEAVES. Unlike legacy preset
    /// limits, this survives source scheduler projection. Decoder bounds still
    /// apply; this cannot increase an existing ceiling or limit fixed trees.
    pub ma_max_leaves: Option<usize>,
    pub split_threshold: f64,
    pub entropy: Coding,
    pub lz77: bool,
    pub lz77_mode: lz77::Lz77Mode,
    /// Exact palette limit; None disables palette analysis.
    pub palette_colors: Option<u32>,
    /// None selects absolute colors; Some selects an exact delta palette.
    pub palette_predictor: Option<Predictor>,
    /// Explicit squeeze passes, true horizontal and false vertical.
    pub squeeze: Vec<bool>,
    pub previous_channels: usize,
    /// Source MA random sampling and pre-quantization, when requested.
    pub sampling: Option<super::sample::SamplingPolicy>,
    /// Optional source channel/wave collection using the caller-owned pool.
    pub source_gather: Option<super::sample::GatherOptions>,
    /// Source JXL_MA_CONSTANT_CHANNELS opt-in; legacy sampling keeps its bypass.
    pub source_constant_channels: bool,
    /// Source opt-in, node-local verified predictor-equivalence reuse.
    pub source_predictor_dedup: bool,
    /// Source opt-in verified equal-partition reuse and inherited property mask.
    pub source_property_dedup: bool,
    /// Exact extra-bit dominance proof for default-scoring MA nodes of weight <=256.
    pub source_ma_dominance: bool,
    /// Explicit source histogram-layout experiments; default leaves them off.
    pub source_ma_histograms: ma_train::HistogramOptions,
    /// Optional source entropy/description-cost experiments; off by default.
    pub source_ma_scoring: ma_train::ScoringOptions,
    /// Optional source statistical/quantile split proposals; off by default.
    pub source_ma_search: ma_train::SearchOptions,
    /// Source whole-subtree switches and shared channel-root execution limits.
    pub source_ma_parallel: ma_train::ParallelOptions,
    /// Source JXL_MA_CHANNEL_TREE, disabled by default. Pixel entry points
    /// require integer main/extra channels of at most eight bits.
    pub source_channel_tree: ma_train::ChannelTreeMode,
    /// Source scheduler gate: exclude disposable probes and tiers above
    /// Tortoise. Direct Modular tool calls have no effort/probe context.
    pub source_channel_tree_final_only: bool,
    /// Source opt-in shared-frame token-channel parallelism; default off.
    pub parallel_tokens: bool,
    /// Hard resource ceiling, not a target or silent truncation of MA samples.
    pub sample_limit: usize,
    pub entropy_options: crate::entropy::options::EntropyOptions,
    /// Score this many source RCT candidates; None uses the explicit rct.
    pub rct_candidates: Option<usize>,
    /// Score modes 0..n; None uses weighted_preset.
    pub weighted_candidates: Option<usize>,
    /// Opt-in preset search parallelism on the caller's installed executor.
    pub parallel_weighted: bool,
    pub fixed_tree: Option<FixedTree>,
    /// Explicit validated MA tree, corresponding to source custom_fixed_tree.
    pub custom_tree: Option<Tree>,
    /// Independent source MA learners, merged using actual stream-ID splits.
    pub tree_groups: Option<super::tree_groups::TreeGroupOptions>,
    /// Empty means one initial domain; otherwise inclusive starts plus final end.
    pub tree_group_splits: Vec<usize>,
    /// Source global/per-group palette eligibility, ordering and cost rejection.
    pub palette_policy: Option<super::palette::PaletteOptions>,
    /// Opt-in Image layout diagnostics. Use a fresh collector for each native
    /// full/probe invocation; cloning options shares this single-use collector.
    /// None does not copy channels, transform scripts or pixel samples.
    pub representation_receipt: Option<std::sync::Arc<receipt::ReceiptCollector>>,
    /// Actual emitted MA tree and entropy-model header bits, not a byte bound.
    /// Use a fresh collector per invocation and read after encoding completes.
    pub model_cost_receipt: Option<std::sync::Arc<receipt_cost::ModelCostReceipt>>,
    /// Executable source LZ parser; length/config fields must match entropy wire.
    pub source_lz77: Option<lz77::SourceConfig>,
    /// Encode the MA tree with source entropy/LZ policy, not legacy prefix-only.
    pub source_tree_entropy: bool,
}
impl Default for ModularOptions {
    fn default() -> Self {
        Self {
            predictor: Predictor::Best,
            rct: Some(6),
            weighted_preset: 0,
            tree_samples: 1024,
            max_tree_leaves: 16,
            max_merged_tree_leaves: None,
            ma_budget_mode: None,
            ma_max_leaves: None,
            split_threshold: 48.0,
            entropy: Coding::Auto,
            lz77: true,
            lz77_mode: lz77::Lz77Mode::Lazy,
            palette_colors: None,
            palette_predictor: None,
            squeeze: Vec::new(),
            previous_channels: 0,
            sampling: None,
            source_gather: None,
            source_constant_channels: false,
            source_predictor_dedup: false,
            source_property_dedup: false,
            source_ma_dominance: false,
            source_ma_histograms: Default::default(),
            source_ma_scoring: Default::default(),
            source_ma_search: Default::default(),
            source_ma_parallel: Default::default(),
            source_channel_tree: Default::default(),
            source_channel_tree_final_only: false,
            parallel_tokens: false,
            sample_limit: 1 << 20,
            entropy_options: Default::default(),
            rct_candidates: None,
            weighted_candidates: None,
            parallel_weighted: false,
            fixed_tree: None,
            custom_tree: None,
            tree_groups: None,
            tree_group_splits: Vec::new(),
            palette_policy: None,
            representation_receipt: None,
            model_cost_receipt: None,
            source_lz77: None,
            source_tree_entropy: false,
        }
    }
}
impl ModularOptions {
    pub(super) fn tree_leaf_limit(&self) -> usize {
        let limit = self.max_merged_tree_leaves.unwrap_or(self.max_tree_leaves);
        self.ma_max_leaves.map_or(limit, |cap| limit.min(cap))
    }

    pub(crate) fn for_pixel_metadata(
        &self,
        metadata: &crate::metadata::ImageMetadata,
    ) -> std::borrow::Cow<'_, Self> {
        let integer8 =
            |depth: crate::metadata::BitDepth| !depth.is_float() && depth.bits_per_sample <= 8;
        if self.source_channel_tree == ma_train::ChannelTreeMode::Disabled
            || (integer8(metadata.bit_depth)
                && metadata
                    .extra_channels
                    .iter()
                    .all(|ec| integer8(ec.bit_depth)))
        {
            std::borrow::Cow::Borrowed(self)
        } else {
            let mut options = self.clone();
            options.source_channel_tree = ma_train::ChannelTreeMode::Disabled;
            std::borrow::Cow::Owned(options)
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.rct.is_some_and(|n| n >= 42)
            || self.weighted_preset > 4
            || self.tree_samples == 0
            || self.tree_samples > 1 << 20
            || self.max_tree_leaves == 0
            || self.max_tree_leaves > ma_train::MAX_LEAVES
            || self
                .max_merged_tree_leaves
                .is_some_and(|limit| !(1..=ma_train::MAX_TREE_LEAVES).contains(&limit))
            || self
                .ma_max_leaves
                .is_some_and(|limit| !(1..=ma_train::MAX_TREE_LEAVES).contains(&limit))
            || !self.split_threshold.is_finite()
            || self.split_threshold < 0.0
        {
            return Err(Error::InvalidInput(
                "invalid Modular tool configuration".into(),
            ));
        }
        self.entropy_options.validate()?;
        self.source_ma_parallel.workers.validate()?;
        if let Some(groups) = &self.tree_groups {
            groups.validate()?;
        }
        if !self.tree_group_splits.is_empty() && self.tree_groups.is_none() {
            return Err(Error::InvalidInput(
                "tree_group_splits requires tree_groups".into(),
            ));
        }
        if self.custom_tree.is_some() && self.fixed_tree.is_some() {
            return Err(Error::InvalidInput(
                "custom_tree and fixed_tree are mutually exclusive".into(),
            ));
        }
        if let Some(config) = &self.source_lz77
            && (config.length_uint_config != self.entropy_options.lz77_length_config
                || config.min_symbol() != self.entropy_options.lz77_min_symbol)
        {
            return Err(Error::InvalidInput(
                "source LZ77 cost/wire configuration must match entropy settings".into(),
            ));
        }
        if let Some(policy) = &self.palette_policy {
            policy.validate()?;
        }
        if let Some(policy) = &self.sampling {
            policy.validate()?;
            if policy.previous_channels != self.previous_channels {
                return Err(Error::InvalidInput(
                    "sampling and coding previous-channel counts differ".into(),
                ));
            }
        }
        if let Some(gather) = &self.source_gather {
            gather.validate()?;
            if self.sampling.is_none() {
                return Err(Error::InvalidInput(
                    "source_gather requires source sampling".into(),
                ));
            }
        }
        if self.sample_limit == 0
            || self.rct_candidates.is_some_and(|n| !(1..=19).contains(&n))
            || self
                .weighted_candidates
                .is_some_and(|n| !(1..=5).contains(&n))
        {
            return Err(Error::InvalidInput(
                "invalid tool search/sample limits".into(),
            ));
        }
        if self.previous_channels > 60 {
            return Err(Error::InvalidInput("previous_channels exceeds 60".into()));
        }
        if self.palette_colors.is_some_and(|n| n == 0 || n > 70000) || self.squeeze.len() > 30 {
            return Err(Error::InvalidInput(
                "palette limit must be 1..=70000; at most 30 squeeze passes".into(),
            ));
        }
        if self.palette_predictor.is_some_and(|p| p.has_weighted())
            || (self.palette_predictor.is_some() && self.palette_colors.is_none())
        {
            return Err(Error::InvalidInput(
                "delta palette needs a color limit and a stateless predictor".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Group {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub stream_id: usize,
}

pub(super) struct Prepared {
    pub planes: Vec<Channel>,
    pub transforms: Vec<Transform>,
    pub wp: weighted::Header,
    pub stream_id: usize,
}

pub(super) fn weighted_header(
    planes: &[Channel],
    options: &ModularOptions,
) -> Result<weighted::Header> {
    let _scope = crate::profile::scope("modular.weighted_header");
    let preset = if let Some(count) = options.weighted_candidates {
        let views: Vec<_> = planes
            .iter()
            .map(|c| super::cost::ChannelView {
                pixels: &c.samples,
                width: c.width,
                height: c.height,
                stride: c.width,
            })
            .collect();
        let selected = if options.parallel_weighted {
            super::cost::select_weighted_preset_parallel(&views, count, true)?
        } else {
            super::cost::select_weighted_preset(&views, count, true)?
        };
        selected.preset
    } else {
        options.weighted_preset
    };
    weighted::Header::preset(u32::from(preset))
}

pub(super) fn prepare(image: &Image, g: Group, options: &ModularOptions) -> Result<Prepared> {
    let _scope = crate::profile::scope("modular.prepare_group");
    let (planes, transforms) = planes(image, g, options)?;
    let wp = weighted_header(&planes, options)?;
    Ok(Prepared {
        planes,
        transforms,
        wp,
        stream_id: g.stream_id,
    })
}

pub(super) fn planes(
    image: &Image,
    g: Group,
    options: &ModularOptions,
) -> Result<(Vec<Channel>, Vec<Transform>)> {
    let mut channels = Vec::with_capacity(image.channel_count());
    for c in 0..image.channel_count() {
        let shift = usize::from(image.channel_shift(c));
        let scale = 1usize << shift;
        let (full_width, _) = image.channel_dimensions(c);
        let x0 = g.x >> shift;
        let y0 = g.y >> shift;
        let width = (g.x + g.width).div_ceil(scale) - x0;
        let height = (g.y + g.height).div_ceil(scale) - y0;
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(width * height)
            .map_err(|_| Error::Limit("Modular channel allocation"))?;
        for y in y0..y0 + height {
            for x in x0..x0 + width {
                let pixel = y * full_width + x;
                samples.push(image.modular_sample(c, pixel));
            }
        }
        channels.push(Channel {
            width,
            height,
            hshift: shift as i32,
            vshift: shift as i32,
            samples,
        });
    }
    let mut transforms = Vec::new();
    if image.color_channels() == 3 {
        let kind = if let Some(count) = options.rct_candidates {
            let views: Vec<_> = channels
                .iter()
                .map(|p| super::cost::ChannelView {
                    pixels: &p.samples,
                    width: p.width,
                    height: p.height,
                    stride: p.width,
                })
                .collect();
            Some(super::cost::select_rct(&views, 0, count)?.rct_type)
        } else {
            options.rct
        };
        if let Some(kind) = kind
            && let Some(transform) = group::apply_rct(&mut channels, 0, 0, kind)?
        {
            transforms.push(transform);
        }
    }
    let mut metadata_channels = 0;
    if let Some(limit) = options.palette_colors
        && let Some(transform) = group::apply_palette(
            &mut channels,
            image.bits(),
            limit,
            options.palette_predictor,
        )?
    {
        transforms.push(transform);
        metadata_channels = 1;
    }
    for &horizontal in &options.squeeze {
        transforms.extend(group::apply_squeeze(
            &mut channels,
            metadata_channels,
            horizontal,
        )?);
        if transforms.len() > 273 {
            return Err(Error::Limit("Modular transform count"));
        }
    }
    Ok((channels, transforms))
}

pub(super) fn predictors(mode: Predictor) -> Vec<Predictor> {
    match mode {
        Predictor::Best => vec![Predictor::Weighted, Predictor::Gradient],
        Predictor::Variable => {
            let mut p: Vec<_> = (0..14)
                .map(|v| Predictor::try_from(v).expect("known predictor"))
                .collect();
            p.swap(0, 6);
            p.swap(1, 5);
            p
        }
        p => vec![p],
    }
}

pub(super) fn visit(
    planes: &[Channel],
    wp: &weighted::Header,
    use_wp: bool,
    previous_channels: usize,
    stream_id: usize,
    mut visitor: impl FnMut(usize, usize, Neighbors, i32, &[i32], i32) -> Result<()>,
) -> Result<()> {
    for c in 0..planes.len() {
        visit_channel(
            planes,
            c,
            wp,
            use_wp,
            previous_channels,
            stream_id,
            &mut visitor,
        )?;
    }
    Ok(())
}

// Each channel reads only original samples; predictor/property history stays
// local, including when the token writer dispatches channels concurrently.
#[allow(clippy::too_many_arguments)]
pub(super) fn visit_channel(
    planes: &[Channel],
    c: usize,
    wp: &weighted::Header,
    use_wp: bool,
    previous_channels: usize,
    stream_id: usize,
    mut visitor: impl FnMut(usize, usize, Neighbors, i32, &[i32], i32) -> Result<()>,
) -> Result<()> {
    let channel = &planes[c];
    let width = channel.width;
    let height = channel.height;
    let plane = &channel.samples;
    if width == 0 || height == 0 {
        return Ok(());
    }
    let references: Vec<_> = planes[..c]
        .iter()
        .rev()
        .filter(|p| {
            p.width == width
                && p.height == height
                && p.hshift == channel.hshift
                && p.vshift == channel.vshift
        })
        .take(previous_channels)
        .collect();
    let mut props = vec![0; 16 + 4 * previous_channels];
    let mut state = if use_wp {
        Some(weighted::State::new(*wp, width)?)
    } else {
        None
    };
    let mut properties = Properties::default();
    for y in 0..height {
        let neighbors = super::predictor::NeighborRow::new(plane, width, width, y)
            .map_err(|e| Error::Encoding(e.to_string()))?;
        for x in 0..width {
            let n = neighbors
                .at(x)
                .map_err(|e| Error::Encoding(e.to_string()))?;
            let pos = y * width + x;
            let (prediction, error) = if let Some(state) = &mut state {
                state.predict_and_update(
                    plane[pos], x, y, n.top, n.left, n.topright, n.topleft, n.toptop,
                )?
            } else {
                (0, 0)
            };
            props[..16].copy_from_slice(&properties.calculate(n, x, y, c, stream_id, error));
            for (i, r) in references.iter().enumerate() {
                let v = r.samples[pos];
                let left = if x > 0 { r.samples[pos - 1] } else { 0 };
                let top = if y > 0 { r.samples[pos - width] } else { left };
                let nw = if x > 0 && y > 0 {
                    r.samples[pos - width - 1]
                } else {
                    left
                };
                let residual = v.wrapping_sub(super::predictor::clamped_gradient(left, top, nw));
                props[16 + 4 * i..20 + 4 * i].copy_from_slice(&[
                    v.wrapping_abs(),
                    v,
                    residual.wrapping_abs(),
                    residual,
                ]);
            }
            visitor(c, pos, n, prediction, &props, plane[pos])?;
        }
    }
    Ok(())
}

pub(super) fn legacy_sample_count(planes: &[Channel], options: &ModularOptions) -> Result<usize> {
    let count = planes.iter().try_fold(0usize, |total, channel| {
        total
            .checked_add(channel.samples.len())
            .ok_or(Error::Limit("legacy MA population"))
    })?;
    Ok(options.tree_samples.min(count))
}

fn legacy_midpoint(next: usize, count: usize, target: usize) -> usize {
    // floor((2 * next + 1) * count / (2 * target)), without usize overflow
    // on 32-bit targets or an extra division for every visited pixel.
    ((next as u128 * count as u128 + count as u128 / 2) / target as u128) as usize
}

pub(super) fn collect_samples(
    planes: &[Channel],
    options: &ModularOptions,
    wp: &weighted::Header,
    stream_id: usize,
) -> Result<Vec<ma_train::Sample>> {
    let target = legacy_sample_count(planes, options)?;
    if target > options.sample_limit {
        return Err(Error::Limit("legacy MA selected-sample budget exceeded"));
    }
    if target == 0 {
        return Ok(Vec::new());
    }
    let count: usize = planes.iter().map(|c| c.samples.len()).sum();
    let predictors = predictors(options.predictor);
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(target)
        .map_err(|_| Error::Limit("MA sample allocation"))?;
    let mut next = 0;
    let mut next_index = legacy_midpoint(next, count, target);
    let mut index = 0;
    visit(
        planes,
        wp,
        predictors.contains(&Predictor::Weighted),
        options.previous_channels,
        stream_id,
        |_, _, n, wp_pred, properties, value| {
            // Distributed midpoint samples include every part of the group instead
            // of taking only its first row. WP state still follows every pixel.
            if next < target && index == next_index {
                let residuals = predictors
                    .iter()
                    .map(|&p| {
                        let guess = predict_one(p, n, Some(i64::from(wp_pred)))
                            .map_err(|e| Error::Encoding(e.to_string()))?;
                        Ok((i64::from(value) - guess) as i32)
                    })
                    .collect::<Result<Vec<_>>>()?;
                samples.push(ma_train::Sample {
                    properties: properties.to_vec(),
                    residuals,
                    weight: 1,
                });
                next += 1;
                if next < target {
                    next_index = legacy_midpoint(next, count, target);
                }
            }
            index += 1;
            Ok(())
        },
    )?;
    Ok(samples)
}

pub(super) fn train_options(options: &ModularOptions) -> ma_train::TrainOptions {
    let mut train = ma_train::TrainOptions {
        channel_tree: options.source_channel_tree,
        predictor_dedup: options.source_predictor_dedup,
        property_dedup: options.source_property_dedup,
        split_dominance: options.source_ma_dominance,
        predictors: predictors(options.predictor),
        properties: (0..16 + 4 * options.previous_channels)
            .filter(|&p| p != 1)
            .collect(),
        max_leaves: options.max_tree_leaves,
        max_merged_leaves: options
            .max_merged_tree_leaves
            .or(options.ma_max_leaves)
            .map(|_| options.tree_leaf_limit()),
        budget_mode: options.ma_budget_mode,
        split_threshold: options.split_threshold,
        ..Default::default()
    };
    options.source_ma_histograms.apply(&mut train);
    options.source_ma_scoring.apply(&mut train);
    options.source_ma_search.apply(&mut train);
    options.source_ma_parallel.apply(&mut train);
    train
}

pub(super) fn predefined<'a>(
    mut planes: impl Iterator<Item = &'a Channel>,
    options: &ModularOptions,
    bits: u8,
) -> Result<Option<Tree>> {
    if let Some(tree) = &options.custom_tree {
        return Ok(Some(tree.clone()));
    }
    let Some(kind) = options.fixed_tree else {
        return Ok(None);
    };
    if kind == FixedTree::GradientLeaf {
        return Tree::channel_predictors(&[Predictor::Gradient]).map(Some);
    }
    let total_pixels = planes.try_fold(0usize, |sum, p| {
        sum.checked_add(p.samples.len())
            .ok_or(Error::Limit("fixed tree population"))
    })?;
    let mode = match kind {
        FixedTree::Zero => super::fixed::FixedTreeMode::Zero,
        FixedTree::Weighted => super::fixed::FixedTreeMode::Weighted { total_pixels },
        FixedTree::Gradient => super::fixed::FixedTreeMode::Gradient {
            total_pixels,
            previous_channels: options.previous_channels,
        },
        FixedTree::GradientLeaf => unreachable!(),
    };
    super::fixed::build(bits, mode).map(Some)
}

fn learn(
    planes: &[Channel],
    options: &ModularOptions,
    wp: &weighted::Header,
    stream_id: usize,
    bits: u8,
) -> Result<Tree> {
    if let Some(tree) = predefined(planes.iter(), options, bits)? {
        return Ok(tree);
    }
    super::training::learn_local(
        &[super::training::TrainingStream {
            channels: planes,
            wp: *wp,
            stream_id,
        }],
        options,
    )
}

pub(super) fn group_stream(
    planes: &[Channel],
    tree: &Tree,
    options: &ModularOptions,
    wp: &weighted::Header,
    stream_id: usize,
) -> Result<(Vec<Token>, bool)> {
    group_stream_with_channels(planes, tree, options, wp, stream_id, false)
}

pub(super) fn group_stream_with_channels(
    planes: &[Channel],
    tree: &Tree,
    options: &ModularOptions,
    wp: &weighted::Header,
    stream_id: usize,
    parallel_channels: bool,
) -> Result<(Vec<Token>, bool)> {
    let distance_multiplier = planes.iter().map(|c| c.width).max().unwrap_or(1);
    let raw = super::tokens::tokenize(
        planes,
        tree,
        wp,
        options.previous_channels,
        stream_id,
        parallel_channels,
    )?;
    if options.lz77
        && let Some(config) = &options.source_lz77
    {
        let _scope = crate::profile::scope("lz77.parse_local");
        let outcome = lz77::parse_source_with_state(&raw, distance_multiplier, config)?;
        let (tokens, _) = commands_to_tokens(outcome.commands, tree.leaves(), distance_multiplier)?;
        return Ok((tokens, outcome.enabled));
    }
    // (4,0,0) length coding reserves symbols 224..255 in this stream.
    let lz_options = Lz77Options {
        mode: options.lz77_mode,
        max_match_length: (1 << 20) + 2,
        ..Default::default()
    };
    let commands = if !options.lz77 {
        raw.iter().copied().map(Command::Literal).collect()
    } else {
        lz77::parse(&raw, distance_multiplier, &lz_options)?
    };
    commands_to_tokens(commands, tree.leaves(), distance_multiplier)
}

pub(crate) fn commands_to_tokens(
    commands: Vec<Command>,
    contexts: usize,
    distance_multiplier: usize,
) -> Result<(Vec<Token>, bool)> {
    let has_matches = commands.iter().any(|t| matches!(t, Command::Match { .. }));
    let mut tokens = Vec::with_capacity(commands.len());
    for command in commands {
        match command {
            Command::Literal(t) => tokens.push(Token::literal(t.context as usize, t.value)),
            Command::Match {
                context,
                length,
                distance,
            } => {
                tokens.push(Token {
                    context: context as usize,
                    value: length
                        .checked_sub(3)
                        .ok_or_else(|| Error::Encoding("invalid LZ77 length".into()))?,
                    is_length: true,
                });
                tokens.push(Token::literal(
                    contexts,
                    lz77::distance_symbol(distance as usize, distance_multiplier)?,
                ));
            }
        }
    }
    Ok((tokens, has_matches))
}

pub(super) fn write_tree(
    tree: &Tree,
    writer: &mut BitWriter,
    options: &ModularOptions,
) -> Result<()> {
    if options.source_tree_entropy {
        if options.lz77 && options.source_lz77.is_none() && options.lz77_mode == lz77::Lz77Mode::Rle
        {
            return tree.write_with_rle(writer, &options.entropy_options);
        }
        tree.write_with_options(
            writer,
            &options.entropy_options,
            if options.lz77 {
                options.source_lz77.as_ref()
            } else {
                None
            },
        )
    } else {
        tree.write(writer)
    }
}

/// Only final output writers call this; model costing and trial tree writes do not.
pub(super) fn write_tree_and_model(
    tree: &Tree,
    model: &Model,
    writer: &mut BitWriter,
    options: &ModularOptions,
) -> Result<()> {
    if let Some(receipt) = &options.model_cost_receipt {
        let before = writer.bits_written();
        write_tree(tree, writer, options)?;
        let after_tree = writer.bits_written();
        model.write_header(writer)?;
        receipt.record(after_tree - before, writer.bits_written() - after_tree);
        Ok(())
    } else {
        write_tree(tree, writer, options)?;
        model.write_header(writer)
    }
}

pub(crate) fn encode_group(
    image: &Image,
    g: Group,
    options: &ModularOptions,
    limit: usize,
    single: bool,
    patches: Option<&crate::tools::patches::PatchDictionary>,
) -> Result<Vec<u8>> {
    let (planes, transforms) = planes(image, g, options)?;
    encode_prepared(
        &planes,
        &transforms,
        options,
        limit,
        single,
        patches,
        g.stream_id,
        image.bits(),
    )
}

// The group script, identity and output framing remain distinct source inputs.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_prepared(
    planes: &[Channel],
    transforms: &[Transform],
    options: &ModularOptions,
    limit: usize,
    global: bool,
    patches: Option<&crate::tools::patches::PatchDictionary>,
    stream_id: usize,
    bits: u8,
) -> Result<Vec<u8>> {
    let mut writer = BitWriter::new(limit);
    if global {
        crate::patch_frame::write_dictionary(&mut writer, patches)?;
        writer.write(1, 1)?;
        writer.write(1, 0)?;
    } // Default DC dequant, no shared tree.
    write_prepared_stream(planes, transforms, options, &mut writer, stream_id, bits)?;
    writer.zero_pad_to_byte()?;
    Ok(writer.into_bytes()?)
}

/// Write a local Modular stream at the caller's current bit position. JPEG
/// quantization/DC/control streams are embedded, not byte-aligned groups.
/// Planes and transform descriptors have already been prepared by the caller;
/// this does not apply the image-level RCT/palette/squeeze plan a second time.
pub(crate) fn write_prepared_stream(
    planes: &[Channel],
    transforms: &[Transform],
    options: &ModularOptions,
    writer: &mut BitWriter,
    stream_id: usize,
    bits: u8,
) -> Result<()> {
    options.validate()?;
    let meta = planes
        .iter()
        .take_while(|p| p.hshift < 0 && p.vshift < 0)
        .count();
    group::validate_layout(planes, meta)?;
    if !(1..=32).contains(&bits) || stream_id > i32::MAX as usize {
        return Err(Error::InvalidInput(
            "invalid local Modular depth/stream ID".into(),
        ));
    }
    for transform in transforms {
        group::validate_transform(transform)?;
    }
    let wp = weighted_header(planes, options)?;
    group::write_header(writer, transforms, &wp, false)?;
    // A zero-entry implicit palette has no coded metadata samples. The decoder
    // returns immediately after GroupHeader when every channel is empty.
    if planes.iter().any(|p| p.width != 0 && p.height != 0) {
        let tree = learn(planes, options, &wp, stream_id, bits)?;
        let (tokens, has_matches) = group_stream(planes, &tree, options, &wp, stream_id)?;
        let streams = [tokens];
        let model = Model::build_with_options(
            &streams,
            tree.leaves() + usize::from(has_matches),
            has_matches,
            options.entropy,
            &options.entropy_options,
        )?;
        write_tree_and_model(&tree, &model, writer, options)?;
        model.write_stream(&streams[0], writer)?;
    }
    Ok(())
}

pub(crate) fn empty_global(
    limit: usize,
    patches: Option<&crate::tools::patches::PatchDictionary>,
) -> Result<Vec<u8>> {
    let mut writer = BitWriter::new(limit);
    crate::patch_frame::write_dictionary(&mut writer, patches)?;
    writer.write(1, 1)?;
    writer.write(1, 0)?;
    group::write_header(&mut writer, &[], &weighted::Header::default(), false)?;
    writer.zero_pad_to_byte()?;
    Ok(writer.into_bytes()?)
}

#[cfg(test)]
mod embedded_stream_tests {
    use super::*;

    fn bits(mut writer: BitWriter) -> Vec<u8> {
        let length = writer.bits_written();
        writer.zero_pad_to_byte().unwrap();
        let bytes = writer.into_bytes().unwrap();
        (0..length).map(|i| (bytes[i / 8] >> (i % 8)) & 1).collect()
    }

    #[test]
    fn embedded_modular_stream_preserves_unaligned_prefix_and_suffix() {
        let channels: Vec<_> = (0..3)
            .map(|c| Channel {
                width: 8,
                height: 8,
                hshift: 0,
                vshift: 0,
                samples: (0..64).map(|i| 1 + ((i * 139 + c * 11) % 65535)).collect(),
            })
            .collect();
        for coding in [Coding::Prefix, Coding::Ans] {
            let options = ModularOptions {
                fixed_tree: Some(FixedTree::GradientLeaf),
                entropy: coding,
                rct: None,
                lz77: false,
                ..Default::default()
            };
            let mut reference = BitWriter::new(1 << 20);
            write_prepared_stream(&channels, &[], &options, &mut reference, 1, 8).unwrap();
            let reference = bits(reference);
            for offset in 0..8 {
                let mut writer = BitWriter::new(1 << 20);
                writer.write(offset, (1 << offset) - 1).unwrap();
                write_prepared_stream(&channels, &[], &options, &mut writer, 1, 8).unwrap();
                assert_eq!(writer.bits_written(), reference.len() + usize::from(offset));
                writer.write(3, 5).unwrap();
                let encoded = bits(writer);
                assert!(encoded[..usize::from(offset)].iter().all(|&bit| bit == 1));
                assert_eq!(&encoded[usize::from(offset)..encoded.len() - 3], reference);
                assert_eq!(&encoded[encoded.len() - 3..], [1, 0, 1]);
            }
        }
    }

    #[test]
    fn invalid_prepared_layout_fails_before_changing_embedded_prefix() {
        let mut channels = vec![Channel {
            width: 1,
            height: 1,
            hshift: 0,
            vshift: 0,
            samples: vec![1],
        }];
        let mut writer = BitWriter::new(100);
        writer.write(3, 5).unwrap();
        let options = ModularOptions::default();
        for depth in [0, 33] {
            assert!(
                write_prepared_stream(&channels, &[], &options, &mut writer, 1, depth).is_err()
            );
            assert_eq!(writer.bits_written(), 3);
        }
        channels[0].width = 2;
        assert!(write_prepared_stream(&channels, &[], &options, &mut writer, 1, 8).is_err());
        assert_eq!(writer.bits_written(), 3);
        assert_eq!(bits(writer), [1, 0, 1]);
    }
}

#[cfg(test)]
mod legacy_sampling_tests {
    use super::*;

    #[test]
    fn midpoint_positions_match_the_exact_formula_and_wide_boundaries() {
        for count in [1, 2, 3, 17, 65536] {
            for target in [1, 2, 17, 65536].into_iter().filter(|&n| n <= count) {
                for next in 0..target {
                    assert_eq!(
                        legacy_midpoint(next, count, target),
                        (((2 * next as u128 + 1) * count as u128) / (2 * target as u128)) as usize
                    );
                }
            }
        }
        assert_eq!(legacy_midpoint(0, usize::MAX, 1), usize::MAX / 2);
        assert_eq!(
            legacy_midpoint(usize::MAX - 1, usize::MAX, usize::MAX),
            usize::MAX - 1
        );
        assert_eq!(legacy_midpoint(32768, 65536, 65536), 32768);
    }

    #[test]
    fn collector_checks_budget_before_predictor_state_or_samples() {
        let planes = [Channel {
            width: 64,
            height: 1,
            hshift: 0,
            vshift: 0,
            samples: (0..64).collect(),
        }];
        let mut options = ModularOptions {
            tree_samples: 64,
            sample_limit: 1,
            ..Default::default()
        };
        assert_eq!(legacy_sample_count(&planes, &options).unwrap(), 64);
        assert!(matches!(
            collect_samples(&planes, &options, &weighted::Header::default(), 0),
            Err(Error::Limit("legacy MA selected-sample budget exceeded"))
        ));
        options.sample_limit = 64;
        assert_eq!(
            collect_samples(&planes, &options, &weighted::Header::default(), 0)
                .unwrap()
                .len(),
            64
        );
    }
}

#[cfg(test)]
mod channel_policy_tests {
    use super::*;
    use crate::color::{ColorEncoding, ColorSpace};
    use crate::metadata::{BitDepth, ExtraChannelInfo, ExtraChannelType, ImageMetadata};

    #[test]
    fn channel_tree_metadata_gate_checks_main_and_every_extra_depth() {
        use ma_train::ChannelTreeMode::{ChannelsAndRowsTwice, Disabled};
        let options = ModularOptions {
            source_channel_tree: ChannelsAndRowsTwice,
            ..Default::default()
        };
        let depths = [
            BitDepth::new(1).unwrap(),
            BitDepth::new(8).unwrap(),
            BitDepth::new(9).unwrap(),
            BitDepth::new(16).unwrap(),
            BitDepth {
                bits_per_sample: 8,
                exponent_bits_per_sample: 3,
            },
            BitDepth {
                bits_per_sample: 16,
                exponent_bits_per_sample: 5,
            },
        ];
        for main in depths {
            for extra in depths {
                let mut metadata =
                    ImageMetadata::new(8, ColorEncoding::srgb(ColorSpace::Rgb)).unwrap();
                metadata.bit_depth = main;
                let mut channel = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
                channel.bit_depth = extra;
                metadata.extra_channels = vec![
                    ExtraChannelInfo::new(ExtraChannelType::Alpha, 8).unwrap(),
                    channel,
                ];
                metadata.validate().unwrap();
                let eligible = !main.is_float()
                    && main.bits_per_sample <= 8
                    && !extra.is_float()
                    && extra.bits_per_sample <= 8;
                let resolved = options.for_pixel_metadata(&metadata);
                assert_eq!(
                    resolved.source_channel_tree,
                    if eligible {
                        ChannelsAndRowsTwice
                    } else {
                        Disabled
                    }
                );
                assert_eq!(matches!(resolved, std::borrow::Cow::Borrowed(_)), eligible);
                assert_eq!(options.source_channel_tree, ChannelsAndRowsTwice);
                assert_eq!(
                    ModularOptions::default()
                        .for_pixel_metadata(&metadata)
                        .source_channel_tree,
                    Disabled
                );
            }
        }
    }
}
