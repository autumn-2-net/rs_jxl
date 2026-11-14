// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Explicit boundary to the currently integrated native encoder. Projection is
//! useful for opt-in native trials, but is NOT source-equivalent when gaps exist.
//! No unsupported palette limit or squeeze script is silently truncated.

use super::params::{CandidateParams, DeltaPalette, LzMethod, Override, TrainingPolicy};
use crate::entropy::options::{
    EntropyOptions, ModularEntropyParams, SpeedTier, UintConfigStrategy,
};
use crate::modular::sample::{SamplingPolicy, SourcePolicy};
use crate::modular::source_options::{
    SourceFrameRoute, SourcePatchReference, SourcePatchReferencePolicy, one_pass_geometry,
};
use crate::modular::{encode::FixedTree, ma_train::MAX_LEAVES};
use crate::modular::{
    palette::{PaletteMode, PaletteOptions},
    predictor::Predictor,
};
use crate::{
    EncodeOptions, Error, Image, Result,
    tools::lz77::{Lz77Mode, SourceConfig, SourceFastPrefix, SourceLz77Method},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityGap {
    AutomaticRctCostSearch,
    /// Legacy report value. Source dictionary entropy and deferred atlas
    /// reference-frame policy are now wired; detector adaptation is separate.
    AutomaticPatchPolicy,
    /// Legacy report value. The source detector now normalizes/subtracts f32
    /// operands and uses source tolerance, quantization and accumulation order.
    PatchDetectorIntegerAdaptation,
    PatchSubsampledExtras,
    PaletteAbove70000,
    PaletteAbove70911,
    PalettePercentAbove100,
    /// Nonempty CandidateParams.squeeze is an explicit LOCAL Rust script,
    /// not source responsive. The separate responsive bool is implemented;
    /// local scripts (alone or combined with responsive) remain unavailable.
    ResponsiveTransformOrchestration,
    ExactDeltaAbove65536,
    GlobalAndChannelPaletteHeuristics,
    PaletteCostRejection,
    ExactZeroDeltaInsertion,
    /// Legacy report value; request counts have no private 30-pass ceiling.
    MoreThanThirtySqueezePasses,
    PerChannelRandomTrainingSamples,
    TrainingPropertyOrderAndQuantiles,
    FixedBackendTrees,
    WeightedPresetSearch,
    EntropyTierPolicy,
    EntropyAlphabetSearch,
    /// Legacy report value. Source training now errors on unfinished splits at
    /// its leaf budget; active resource bounds are in NativeResourceLimits.
    NativeTreeLeafLimit,
    TreeEntropyPolicy,
    ForcedPatchesOff,
    /// Diagnostic probe-only palette forcing, not a matched whole-image representation.
    ForcedProbePalette,
    /// Experimental full/probe literal palette tools, not source defaults.
    LiteralPaletteTools,
    /// Probe ranking amortizes measured tree/model bits; never an encoded-byte bound.
    ModelScaledProbeScore,
    /// Probe-only reduced MA learning; complete candidate policy is unchanged.
    ReducedProbeTrainingSamples,
    PaletteRctTransformOrder,
    LzMethodConfiguration,
    LzEntropyWireAlignment,
    /// Legacy report value. Tree, local and shared writers now retain the
    /// source parser's accepted-result flag independently of Match commands.
    LzEnabledState,
    CarefulFirstPass,
    BucketMatcherSelection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptationState {
    /// All requested representation and policy settings are implemented.
    SourceEquivalent,
    /// Usable native options, but the listed source-policy differences remain.
    NativeProxy,
    /// At least one representation cannot be expressed; no encode options.
    Unavailable,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeResourcePolicy {
    /// Explicit whole-tree ceiling, including Some(MAX_LEAVES). None selects
    /// the source global decoder bound, never the direct encoder's base preset.
    /// An unfinished search is an error, not permission to truncate the tree.
    pub max_tree_leaves: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeResourceLimits {
    /// Legacy learner preset (or the explicit caller ceiling).
    pub max_tree_leaves: usize,
    /// Effective whole-tree ceiling after intersecting caller and decoder bounds.
    pub max_merged_tree_leaves: usize,
    /// Optional source gathering ceiling inherited from base.modular.sample_limit.
    pub sample_limit: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct NativeProjection {
    /// None means the representation itself cannot currently be expressed.
    /// Some with gaps is an explicitly approximate native trial, not an exact
    /// C++ candidate; never reuse its score under a source-equivalent cache key.
    pub options: Option<EncodeOptions>,
    pub gaps: Vec<CapabilityGap>,
    pub training: TrainingPolicy,
    pub lz_method: LzMethod,
    pub resources: NativeResourceLimits,
}
impl NativeProjection {
    pub fn state(&self) -> AdaptationState {
        if self.options.is_none() {
            AdaptationState::Unavailable
        } else if self.gaps.is_empty() {
            AdaptationState::SourceEquivalent
        } else {
            AdaptationState::NativeProxy
        }
    }
    pub fn exact_options(self) -> Result<EncodeOptions> {
        if !self.gaps.is_empty() {
            return Err(Error::Encoding(format!(
                "unported scheduler capabilities: {:?}",
                self.gaps
            )));
        }
        self.options
            .ok_or_else(|| Error::Encoding("candidate representation unavailable".into()))
    }
}

/// `enc_frame.cc::GetGroupSizeShift`, pixel lossless, decoding tier zero.
pub fn automatic_group_size_shift(width: u32, height: u32) -> u8 {
    crate::modular::source_options::automatic_group_size_shift(width, height)
}

/// Build a callback-free policy from ORIGINAL candidate inputs. Only reference
/// overrides are applied before selecting predictor-dependent LZ/WP settings.
/// The image supplies original metadata, not final reference geometry. The
/// backend calls resolve on its signed atlas without depending on scheduler.
pub fn source_patch_reference_policy(
    params: &CandidateParams,
    image: &Image,
    base: &EncodeOptions,
    resources: NativeResourcePolicy,
) -> Result<SourcePatchReferencePolicy> {
    let mut reference = params.clone();
    reference.max_properties =
        Some(params.previous_channel_budget(base.modular.source_tool_defaults) as u8);
    // RoundtripPatchFrame receives the post-frame-fallback speed, but resets
    // responsive to 0 before selecting the atlas's predictor/transforms/tree.
    reference.speed_tier = params.modular_speed_tier();
    reference.responsive = false;
    reference.predictor = Some(Predictor::Gradient);
    reference.patches = Override::Off;
    // Explicit Rust squeeze passes are not the source responsive switch and
    // remain unavailable, rather than being silently erased here.
    let options = project_native_metadata(&reference, image, base, resources)?.exact_options()?;
    let mut modular = options.modular;
    modular.sampling = None;
    let policy = SourcePatchReferencePolicy {
        modular,
        group_size_shift: params.group_size_shift,
        buffering: params.buffering,
        sampling: SourcePolicy {
            speed_tier: reference.speed_tier,
            predictor: Predictor::Gradient,
            tree_mode: params.tree_mode,
            learning_fraction: params.learning_fraction,
            max_previous_channels: params
                .previous_channel_budget(base.modular.source_tool_defaults),
            previous_property_subset: params.previous_property_subset,
            num_extra_channels: image.metadata().extra_channels.len(),
            is_gray: image.color_channels() == 1,
            ..Default::default()
        },
    };
    // Metadata-only validation also accepts signed working atlases, whose
    // external PixelData/extra_planes storage is deliberately empty.
    policy.resolve(image)?;
    Ok(policy)
}

/// Direct projection when the actual atlas is already available. This is the
/// same typed policy that the backend resolves later, not a main-options clone.
pub fn project_source_patch_reference(
    params: &CandidateParams,
    atlas: &Image,
    base: &EncodeOptions,
    resources: NativeResourcePolicy,
) -> Result<SourcePatchReference> {
    source_patch_reference_policy(params, atlas, base, resources)?.resolve(atlas)
}

pub fn project_native(
    params: &CandidateParams,
    image: &Image,
    base: &EncodeOptions,
) -> Result<NativeProjection> {
    project_native_with_resources(params, image, base, NativeResourcePolicy::default())
}

pub fn project_native_with_resources(
    params: &CandidateParams,
    image: &Image,
    base: &EncodeOptions,
    resources: NativeResourcePolicy,
) -> Result<NativeProjection> {
    // EncodeLosslessSearch's <=16-bit integer gate belongs to the automatic
    // driver, not a direct Modular candidate (nor the e11 frame tournament).
    // Image's immutable constructors already validate typed sample storage,
    // including lossless F32 -> custom-float conversion and U31 ranges.
    project_native_metadata(params, image, base, resources)
}

// Shared settings projection: reads geometry, channel counts and metadata
// only. Never inspect external PixelData here: a private atlas stores its valid
// working words separately and intentionally exposes empty external buffers.
fn project_native_metadata(
    params: &CandidateParams,
    image: &Image,
    base: &EncodeOptions,
    resources: NativeResourcePolicy,
) -> Result<NativeProjection> {
    // Resolve auto before deriving both the tree property list and sampling.
    // Explicit zero must remain distinguishable from the e9 prev1 default.
    let mut resolved_params = params.clone();
    resolved_params.max_properties =
        Some(params.previous_channel_budget(base.modular.source_tool_defaults) as u8);
    let params = &resolved_params;
    params.validate()?;
    image.metadata().validate()?;
    let resource_policy = resources;
    let decoder_leaves = crate::modular::ma_train::global_tree_leaf_limit(
        image.width() as usize,
        image.height() as usize,
        image.channel_count(),
    );
    let resources = NativeResourceLimits {
        max_tree_leaves: resources.max_tree_leaves.unwrap_or(MAX_LEAVES),
        max_merged_tree_leaves: resources
            .max_tree_leaves
            .map_or(decoder_leaves, |limit| limit.min(decoder_leaves)),
        sample_limit: base.modular.sample_limit,
    };
    if !(1..=MAX_LEAVES).contains(&resources.max_tree_leaves) {
        return Err(Error::InvalidInput(
            "native scheduler leaf resource bound outside 1..=MAX_LEAVES".into(),
        ));
    }
    let mut options = base.clone();
    if let std::borrow::Cow::Owned(resolved) = options.modular.for_pixel_metadata(image.metadata())
    {
        options.modular = resolved;
    }
    options.source_patch_reference = None;
    let speed_tier = params.modular_speed_tier();
    let geometry = one_pass_geometry(image.width(), image.height(), params.group_size_shift)?;
    let route = geometry.pixel_route(
        params.buffering,
        speed_tier,
        params.patches == Override::On,
        params.responsive,
    );
    let streaming = route == SourceFrameRoute::StreamingFrameAreas;
    options.group_size_shift = geometry.group_size_shift;
    // Frame-area orchestration independently encodes its local streams and
    // ignores shared here. Keep the correct shared-tree full-frame fallback;
    // shared_tree=false alone would not implement source streaming geometry.
    options.shared_tree = route != SourceFrameRoute::IndependentFullFrame;
    let parallel_shared_e5 =
        speed_tier == 5 && !streaming && options.shared_tree && base.threads > 1;
    // Source responsive alone does not enable progressive passes. Never
    // inherit an unrelated base frame layout into an explicit source candidate.
    options.responsive = params.responsive;
    options.passes = Default::default();
    let dim = geometry.group_dim;
    let streams = geometry.num_streams;
    let training = params.training_policy(image.channel_count(), streams);
    let lz_method = params.lz_method();
    // Local and shared writers route through encode::write_tree, which honors
    // source_tree_entropy and Tree::write_with_options for this candidate.
    let mut gaps = Vec::new();
    let mut representable = true;
    options.modular.predictor = params.resolved_predictor();
    // Source automatic search excludes custom_fixed_tree. A direct-encode
    // override in base must not replace this candidate's learned/fixed policy.
    options.modular.custom_tree = None;
    // Source-default candidates do not request MA shard/split extensions.
    // Clear both together before validation, including orphaned base splits.
    options.modular.tree_groups = None;
    options.modular.tree_group_splits.clear();
    // JXL_MA_CONSTANT_CHANNELS is opt-in, not a source-default optimization.
    options.modular.source_constant_channels = false;
    // Exact MA search proofs (candidate 4, equivalent to CLI --ma-prune):
    // property-partition reuse plus small-node dominance pruning only skip
    // search visits; the chosen splits, budget and codestream stay identical.
    // Forced on for every scheduler candidate; an explicit base value is
    // overridden because these are proofs, not an approximation knob.
    options.modular.source_property_dedup = true;
    options.modular.source_ma_dominance = true;
    // Candidate N3 (2026-11-11): SIMD-aligned cache-set-avoiding strides for
    // the property histogram fill (source JXL_MA_PADDED_HISTOGRAM=1).
    // Measured -4.6% against ma-cache2k in one pass, but two cumulative passes
    // against ma-prune were 0.968x / 0.907x: the padded stride enlarges every
    // cache that N1 admits, so the stacked pair is net-negative. Reverted.
    // (Candidate N2 histogram_shards measured 0.861x and was reverted too.)
    if options.modular.source_channel_tree_final_only && speed_tier > 1 {
        options.modular.source_channel_tree = crate::modular::ma_train::ChannelTreeMode::Disabled;
    }
    // ComputeTree ignores the explicit predictor for these fixed speed tiers.
    // Tier 9 is a SINGLE Gradient leaf, not kGradientFixedDC.
    options.modular.fixed_tree = match speed_tier {
        7 => Some(FixedTree::Weighted),
        8 => Some(FixedTree::Gradient),
        9 => Some(FixedTree::GradientLeaf),
        _ => None,
    };
    // Explicit tree experiments replace the low-effort fixed tree, not the
    // effort's transform, entropy, LZ77 or sampling settings.
    if options.modular.fast_context.is_some() || options.modular.source_ma_seed.is_some() {
        options.modular.fixed_tree = None;
    }
    // Native e9 defaults only: explicit tree plans and alternative learners keep
    // their own policy, including plans cleared by source projection above.
    let requested = &base.modular;
    if requested.source_ma_defaults
        && speed_tier == 1
        && requested.fixed_tree.is_none()
        && requested.custom_tree.is_none()
        && requested.fast_context.is_none()
        && requested.block_context.is_none()
        && requested.source_ma_seed.is_none()
        && requested.source_ma_lookahead.is_none()
        && !requested.tree_learning.enabled()
        && requested.tree_groups.is_none()
        && requested.tree_group_splits.is_empty()
        && requested.source_channel_tree == crate::modular::ma_train::ChannelTreeMode::Disabled
        && requested.source_ma_search == Default::default()
    {
        use crate::modular::ma_train::{DescriptionCost, TreeLearning};
        use crate::modular::refinement::{OffsetMode, OffsetOptions};

        options
            .modular
            .refinement
            .offsets
            .get_or_insert_with(|| OffsetOptions {
                mode: OffsetMode::DistributionSinglePass,
                ..Default::default()
            });
        if requested.source_ma_scoring.description == DescriptionCost::Disabled {
            options.modular.source_ma_scoring.description = DescriptionCost::Half;
        }
        // Explicit alternative entropy estimators keep their existing learner.
        // Symmetric includes description cost for all three child regions.
        if requested.source_ma_scoring.description_only() {
            options.modular.tree_learning = TreeLearning::Symmetric;
        }
    }
    // training::learn enables require_complete_search whenever sampling is
    // Some. Budget exhaustion with beneficial splits errors, never truncates.
    options.modular.max_tree_leaves = resources.max_tree_leaves;
    options.modular.max_merged_tree_leaves = Some(resources.max_merged_tree_leaves);
    options.modular.rct_candidates = None;
    if let Some(rct) = params.rct {
        options.modular.rct = if rct == 0 { None } else { Some(rct) };
    } else if image.channel_count() >= 3 {
        if speed_tier > 5 || params.responsive {
            options.modular.rct = Some(6);
        } else {
            options.modular.rct = None;
            options.modular.rct_candidates = Some(params.automatic_rcts().len());
        }
    } else {
        options.modular.rct = None;
    }
    options.modular.previous_channels = training.previous_channels;
    options.modular.sampling = Some(SamplingPolicy::from_source(&SourcePolicy {
        speed_tier,
        predictor: options.modular.predictor,
        tree_mode: params.tree_mode,
        num_streams: streams,
        responsive: params.responsive,
        learning_fraction: params.learning_fraction,
        max_previous_channels: params.previous_channel_budget(base.modular.source_tool_defaults),
        previous_property_subset: params.previous_property_subset,
        num_extra_channels: image.metadata().extra_channels.len(),
        is_gray: image.color_channels() == 1,
        max_channel_size: dim,
        ..Default::default()
    })?);
    if parallel_shared_e5 {
        // Shared domains have no outer frame-area jobs to distribute this work.
        // Keep complete channel histories and sampling policy; only dispatch changes.
        options.modular.source_gather.get_or_insert_with(|| {
            crate::modular::sample::GatherOptions {
                workers: base.threads.clamp(1, 64),
                ..Default::default()
            }
        });
    }
    // Keep sample_limit as the caller's hard error ceiling; do not silently
    // reduce the source sampling fraction to fit a midpoint target.
    options.modular.entropy_options = EntropyOptions::for_modular(ModularEntropyParams {
        speed_tier: SpeedTier::try_from(speed_tier)?,
        precise_entropy: params.precise_entropy,
        ..Default::default()
    });
    options.modular.entropy_options.streaming_mode = streaming;
    options.modular.entropy = crate::entropy::stream::Coding::Source;
    options.modular.source_tree_entropy = true;
    options.modular.weighted_preset = 0;
    options.modular.weighted_candidates =
        (params.weighted_modes() > 1).then_some(usize::from(params.weighted_modes()));
    options.modular.squeeze = params.squeeze.clone();
    // The source policy owns both global/local stages and overrides legacy
    // all-channel palette fields, including when only channel compaction is on.
    options.modular.palette_colors = None;
    options.modular.palette_predictor = None;
    let mut palette_tools = options
        .modular
        .palette_policy
        .as_ref()
        .map(|p| p.tools)
        .unwrap_or_default();
    if speed_tier == 1 && base.modular.source_tool_defaults && base.modular.palette_policy.is_none()
    {
        palette_tools.order = crate::modular::palette::PaletteOrder::ScreenedRuns;
    }
    let palette_grouping = options
        .modular
        .palette_policy
        .as_ref()
        .and_then(|p| p.grouping);
    // JXL_LOSSLESS_PALETTE_FORCE is available as an explicit base-policy
    // override, not a predictor/default change. Keep its source color-cap gate.
    let force_palette = (1..=65536).contains(&params.palette_colors)
        && base
            .modular
            .palette_policy
            .as_ref()
            .is_some_and(|p| p.force_palette);
    options.modular.palette_policy = None;
    options.global_palette = true;
    if params.palette_colors > 70911 {
        gaps.push(CapabilityGap::PaletteAbove70911);
        representable = false;
    }
    if params.channel_colors_percent > 100.0 || params.channel_colors_pre_transform_percent > 100.0
    {
        gaps.push(CapabilityGap::PalettePercentAbove100);
        representable = false;
    }
    if !params.squeeze.is_empty() {
        // The source global script is selected ONLY by params.responsive.
        // A local direction list is neither equivalent to that script nor a
        // source CompressParams setting. Do not silently erase or relocate it.
        gaps.push(CapabilityGap::ResponsiveTransformOrchestration);
        representable = false;
    }
    if params.palette_colors <= 70911 {
        let mode = match params.delta_palette {
            DeltaPalette::Disabled => PaletteMode::Absolute,
            DeltaPalette::ExactGradient => PaletteMode::ExactDelta(Predictor::Gradient),
            DeltaPalette::ExactLeft => PaletteMode::ExactDelta(Predictor::Left),
            DeltaPalette::ExactTop => PaletteMode::ExactDelta(Predictor::Top),
            DeltaPalette::ZeroGradient => PaletteMode::ExactZero(Predictor::Gradient),
            DeltaPalette::ZeroLeft => PaletteMode::ExactZero(Predictor::Left),
            DeltaPalette::ZeroTop => PaletteMode::ExactZero(Predictor::Top),
        };
        // The exact-delta ATTEMPT still requires <=65536 colors. The source
        // policy first computes the actual per-attempt cap, then falls back to
        // an absolute palette if delta declines. A higher policy cap is valid.
        options.modular.palette_policy = Some(PaletteOptions {
            palette_colors: params.palette_colors as i32,
            channel_colors_pre_transform_percent: params.channel_colors_pre_transform_percent,
            channel_colors_percent: params.channel_colors_percent,
            speed_tier,
            responsive: params.responsive,
            groupwise: streaming,
            do_color: true,
            force_palette,
            mode,
            tools: palette_tools,
            grouping: palette_grouping,
            ..Default::default()
        });
    }
    // enc_modular.cc::ComputeEncodingData's OUTER gate also constrains On.
    // The source non-XYB fast detector's On/Default routes both require seeds.
    // Legacy Replace patches are never a substitute for source Add patches.
    options.patches = None;
    options.tile_patches = None;
    options.source_patches = None;
    if params.patches != Override::Off
        && crate::tools::patches::source_enabled(
            u32::from(image.bits()),
            speed_tier,
            0,
            true,
            streaming,
        )
    {
        let config = crate::tools::patches::SourcePatchConfig {
            speed_tier,
            // FindTextLikePatches observes COLOR only. Extra planes have None
            // blending and do not change integer color detection/conversion.
            floating_point: image.metadata().bit_depth.is_float(),
            extra_channel_dim_shifts: image
                .metadata()
                .extra_channels
                .iter()
                .map(|ec| u32::from(ec.dim_shift))
                .collect(),
            ..Default::default()
        };
        if !crate::tools::patches::source_supported(
            u32::from(image.bits()),
            config.floating_point,
            &config.extra_channel_dim_shifts,
        ) {
            gaps.push(CapabilityGap::PatchSubsampledExtras);
            representable = false;
        }
        options.source_patches = Some(config);
    }
    options.modular.lz77 = lz_method != LzMethod::None;
    options.modular.source_lz77 = match lz_method {
        LzMethod::None => None,
        method => Some(SourceConfig {
            method: match method {
                LzMethod::Lz77b1w3t => SourceLz77Method::Lz77b1w3t,
                LzMethod::Lz77b3w3t => SourceLz77Method::Lz77b3w3t,
                LzMethod::Optc8 => SourceLz77Method::Optc8,
                LzMethod::Optc256 => SourceLz77Method::Optc256,
                LzMethod::None => unreachable!("disabled source LZ77 handled above"),
            },
            bucket_matcher: params.bucket_matcher,
            careful_first_pass: params.lz77_careful_first_pass,
            // The frozen source default leaves the fast spatial prefix cache
            // off. Its three-token tag is an exact necessary-condition test, so
            // the lazy SIMD cache only skips candidates whose mandatory prefix
            // cannot match: match order, lengths, distances, symbols and the
            // codestream are unchanged. An explicit caller choice still wins.
            fast_prefix: base
                .modular
                .source_lz77
                .as_ref()
                .and_then(|config| config.fast_prefix)
                .or(Some(SourceFastPrefix::Simd)),
            literal_uint_config: match options.modular.entropy_options.uint_config {
                UintConfigStrategy::Fixed(config) => config,
                UintConfigStrategy::Fast | UintConfigStrategy::Best => Default::default(),
            },
            length_uint_config: options.modular.entropy_options.lz77_length_config,
            min_symbol: Some(options.modular.entropy_options.lz77_min_symbol),
            // Explicit Rust tool switches replace source process-wide knobs.
            // Keep those across candidates; only the candidate-owned parser,
            // seed/bucket parameters and actual wire model are overridden here.
            ..base
                .modular
                .source_lz77
                .clone()
                .unwrap_or_else(|| SourceConfig {
                    quick: if speed_tier == 1 && base.modular.source_tool_defaults {
                        crate::tools::lz77::SourceQuickMode::Sparse
                    } else {
                        crate::tools::lz77::SourceQuickMode::Off
                    },
                    chain_batch: true,
                    chain_batch_parallel_only: true,
                    // Frame-area jobs already run in parallel. Shared e5 needs
                    // stream dispatch here; explicit caller policies still win.
                    parallelism: (method == LzMethod::Optc256 || parallel_shared_e5).then_some(
                        crate::tools::lz77::SourceParallelism {
                            max_workers: base.threads.clamp(1, 64),
                            fast_streams: parallel_shared_e5,
                            ..Default::default()
                        },
                    ),
                    ..Default::default()
                })
        }),
    };
    // Kept coherent for diagnostic consumers. source_lz77 is the active path;
    // b3 is a GREEDY three-entry ring, never the legacy lazy parser.
    options.modular.lz77_mode = match lz_method {
        LzMethod::None | LzMethod::Lz77b1w3t | LzMethod::Lz77b3w3t => Lz77Mode::Greedy,
        LzMethod::Optc8 | LzMethod::Optc256 => Lz77Mode::Optimal,
    };
    options.modular.split_threshold = f64::from(training.node_threshold);
    if representable {
        options.modular.validate()?;
        if options.source_patches.is_some() {
            options.source_patch_reference = Some(source_patch_reference_policy(
                params,
                image,
                base,
                resource_policy,
            )?);
        }
    }
    Ok(NativeProjection {
        options: representable.then_some(options),
        gaps,
        training,
        lz_method,
        resources,
    })
}

#[cfg(test)]
mod tectonic_tests;

#[cfg(test)]
mod direct_input_tests;

#[cfg(test)]
mod responsive_tests;

#[cfg(test)]
mod source_ma_defaults_tests;

#[cfg(test)]
mod source_tool_defaults_tests;
