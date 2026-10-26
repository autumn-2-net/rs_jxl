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
    tools::lz77::{Lz77Mode, SourceConfig, SourceLz77Method},
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
    /// Source gathering hard ceiling inherited from base.modular.sample_limit.
    pub sample_limit: usize,
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
            max_previous_channels: usize::from(params.max_properties),
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
        max_previous_channels: usize::from(params.max_properties),
        num_extra_channels: image.metadata().extra_channels.len(),
        is_gray: image.color_channels() == 1,
        max_channel_size: dim,
        ..Default::default()
    })?);
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
    if params.squeeze.len() > 30 {
        gaps.push(CapabilityGap::MoreThanThirtySqueezePasses);
        representable = false;
    }
    // The source policy owns both global/local stages and overrides legacy
    // all-channel palette fields, including when only channel compaction is on.
    options.modular.palette_colors = None;
    options.modular.palette_predictor = None;
    let palette_tools = options
        .modular
        .palette_policy
        .as_ref()
        .map(|p| p.tools)
        .unwrap_or_default();
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
            ..Default::default()
        });
    }
    // enc_modular.cc::ComputeEncodingData's OUTER gate also constrains On.
    // The source non-XYB fast detector's On/Default routes both require seeds.
    // Legacy Replace patches are never a substitute for source Add patches.
    options.patches = None;
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
            literal_uint_config: match options.modular.entropy_options.uint_config {
                UintConfigStrategy::Fixed(config) => config,
                UintConfigStrategy::Fast | UintConfigStrategy::Best => Default::default(),
            },
            length_uint_config: options.modular.entropy_options.lz77_length_config,
            min_symbol: Some(options.modular.entropy_options.lz77_min_symbol),
            // Explicit Rust tool switches replace source process-wide knobs.
            // Keep those across candidates; only the candidate-owned parser,
            // seed/bucket parameters and actual wire model are overridden here.
            ..base.modular.source_lz77.clone().unwrap_or_default()
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
mod tectonic_tests {
    use super::*;
    use crate::modular::source_options::SourceBuffering;
    use crate::{PixelData, PixelLayout};

    #[test]
    fn palette_tools_survive_candidate_and_reference_projection() {
        use crate::modular::palette::{
            PaletteGeometry, PaletteMapping, PaletteOrder, PaletteTools,
        };
        let image = Image::new(8, 8, PixelLayout::Rgb, PixelData::U8(vec![19; 8 * 8 * 3])).unwrap();
        for mode in 0..=9 {
            let tools = PaletteTools {
                order: PaletteOrder::from_source_mode(mode).unwrap(),
                geometry: PaletteGeometry::from_source_mode(mode % 3).unwrap(),
                mapping: PaletteMapping::Flat,
            };
            let mut base = EncodeOptions::default();
            base.modular.palette_policy = Some(PaletteOptions {
                tools,
                ..Default::default()
            });
            for effort in [3, 9, 11] {
                let params = CandidateParams::default().backend(effort).unwrap();
                let options = project_native(&params, &image, &base)
                    .unwrap()
                    .exact_options()
                    .unwrap();
                assert_eq!(
                    options.modular.palette_policy.as_ref().unwrap().tools,
                    tools
                );
                let atlas = project_source_patch_reference(
                    &params,
                    &image,
                    &base,
                    NativeResourcePolicy::default(),
                )
                .unwrap();
                assert_eq!(atlas.modular.palette_policy.as_ref().unwrap().tools, tools);
            }
        }
    }

    #[test]
    fn explicit_force_palette_survives_candidate_and_reference_projection() {
        let image = Image::new(8, 8, PixelLayout::Rgb, PixelData::U8(vec![19; 8 * 8 * 3])).unwrap();
        let mut base = EncodeOptions::default();
        base.modular.palette_policy = Some(PaletteOptions {
            force_palette: true,
            ..Default::default()
        });
        let original_policy = base.modular.palette_policy.clone();
        for palette_colors in [1, 10000, 65536] {
            for predictor in [None, Some(Predictor::Zero)] {
                let params = CandidateParams {
                    speed_tier: 1,
                    palette_colors,
                    predictor,
                    ..Default::default()
                };
                let original_params = params.clone();
                for modular in [
                    project_native(&params, &image, &base)
                        .unwrap()
                        .exact_options()
                        .unwrap()
                        .modular,
                    project_source_patch_reference(
                        &params,
                        &image,
                        &base,
                        NativeResourcePolicy::default(),
                    )
                    .unwrap()
                    .modular,
                ] {
                    let policy = modular.palette_policy.unwrap();
                    assert!(policy.force_palette);
                    assert_eq!(policy.palette_colors, palette_colors as i32);
                    assert!(!policy.zero_tokens);
                }
                assert_eq!(params, original_params);
                assert_eq!(base.modular.palette_policy, original_policy);
            }
        }
    }

    #[test]
    fn force_palette_defaults_and_outside_caps_do_not_enable_override() {
        let image = Image::new(8, 8, PixelLayout::Rgb, PixelData::U8(vec![19; 8 * 8 * 3])).unwrap();
        for (force_palette, palette_colors) in [
            (None, 10000),
            (Some(false), 10000),
            (Some(true), 0),
            (Some(true), 65537),
            (Some(true), 70000),
            (Some(true), 70911),
        ] {
            let mut base = EncodeOptions::default();
            base.modular.palette_policy = force_palette.map(|force_palette| PaletteOptions {
                force_palette,
                ..Default::default()
            });
            let original_policy = base.modular.palette_policy.clone();
            let params = CandidateParams {
                speed_tier: 1,
                palette_colors,
                predictor: Some(Predictor::Zero),
                ..Default::default()
            };
            let original_params = params.clone();
            for modular in [
                project_native(&params, &image, &base)
                    .unwrap()
                    .exact_options()
                    .unwrap()
                    .modular,
                project_source_patch_reference(
                    &params,
                    &image,
                    &base,
                    NativeResourcePolicy::default(),
                )
                .unwrap()
                .modular,
            ] {
                let policy = modular.palette_policy.unwrap();
                assert!(!policy.force_palette);
                assert_eq!(policy.palette_colors, palette_colors as i32);
                assert!(!policy.zero_tokens);
            }
            assert_eq!(params, original_params);
            assert_eq!(base.modular.palette_policy, original_policy);
        }
    }

    #[test]
    fn explicit_lz_tools_survive_candidates_without_overriding_wire_policy() {
        use crate::tools::lz77::{SourceIndexMode, SourceParallelism};
        let image = Image::new(
            32,
            32,
            PixelLayout::Rgb,
            PixelData::U8(vec![19; 32 * 32 * 3]),
        )
        .unwrap();
        let mut base = EncodeOptions::default();
        let parallelism = Some(SourceParallelism {
            max_workers: 3,
            ..Default::default()
        });
        base.modular.source_lz77 = Some(SourceConfig {
            stream_arena: true,
            stream_arena_quiet: true,
            index_mode: SourceIndexMode::All,
            parallelism,
            index_word_match: true,
            min_symbol: Some(99),
            ..Default::default()
        });
        for effort in [5, 9, 11] {
            let mut params = CandidateParams::default().backend(effort).unwrap();
            params.predictor = Some(Predictor::Zero);
            for options in [
                project_native(&params, &image, &base)
                    .unwrap()
                    .exact_options()
                    .unwrap()
                    .modular,
                project_source_patch_reference(
                    &params,
                    &image,
                    &base,
                    NativeResourcePolicy::default(),
                )
                .unwrap()
                .modular,
            ] {
                let lz = options.source_lz77.as_ref().unwrap();
                assert!(lz.stream_arena && lz.stream_arena_quiet && lz.index_word_match);
                assert_eq!(lz.parallelism, parallelism);
                assert_eq!(lz.index_mode, SourceIndexMode::All);
                assert_eq!(
                    lz.length_uint_config,
                    options.entropy_options.lz77_length_config
                );
                assert_eq!(lz.min_symbol(), options.entropy_options.lz77_min_symbol);
                assert_ne!(lz.min_symbol(), 99);
            }
        }
        let params = CandidateParams {
            speed_tier: 7,
            ..Default::default()
        };
        let options = project_native(&params, &image, &base)
            .unwrap()
            .exact_options()
            .unwrap();
        assert!(options.modular.source_lz77.is_none());
        assert!(!options.modular.lz77);
    }

    #[test]
    fn negative_tier_reaches_sampling_palette_entropy_lz_and_patch_reference() {
        let image =
            Image::new(2304, 1, PixelLayout::Rgb, PixelData::U8(vec![7; 2304 * 3])).unwrap();
        let params = CandidateParams::default().backend(11).unwrap();
        let projection = project_native(&params, &image, &EncodeOptions::default()).unwrap();
        assert_eq!(projection.training.node_threshold, 61);
        assert_eq!(projection.lz_method, LzMethod::Optc256);
        let options = projection.exact_options().unwrap();
        assert!(options.shared_tree && options.global_palette);
        assert_eq!(options.modular.predictor, Predictor::Variable);
        assert!(options.modular.fixed_tree.is_none());
        assert_eq!(options.modular.rct_candidates, Some(19));
        assert_eq!(options.modular.weighted_candidates, Some(5));
        let palette = options.modular.palette_policy.as_ref().unwrap();
        assert_eq!(palette.speed_tier, -1);
        assert!(!palette.groupwise);
        let sampling = options.modular.sampling.as_ref().unwrap();
        assert_eq!(sampling.properties, vec![0, 1, 15]);
        assert_eq!(sampling.max_property_values, 32);
        assert_eq!(sampling.node_threshold, 61.0);
        assert_eq!(sampling.nb_repeats, 0.5f32 * 0.3f32);
        assert_eq!(
            options.modular.entropy_options,
            EntropyOptions::for_modular(ModularEntropyParams {
                speed_tier: SpeedTier::TectonicPlate,
                ..Default::default()
            })
        );
        let lz = options.modular.source_lz77.as_ref().unwrap();
        assert_eq!(lz.method, SourceLz77Method::Optc256);
        assert_eq!(
            lz.length_uint_config,
            options.modular.entropy_options.lz77_length_config
        );
        assert_eq!(
            lz.min_symbol(),
            options.modular.entropy_options.lz77_min_symbol
        );
        assert_eq!(options.source_patches.as_ref().unwrap().speed_tier, -1);
        let recipe = options.source_patch_reference.unwrap();
        assert_eq!(recipe.sampling.speed_tier, -1);
        let atlas = image
            .with_modular_planes(16, 1, vec![vec![-1; 16]; 3])
            .unwrap();
        let reference = recipe.resolve(&atlas).unwrap();
        assert_eq!(reference.modular.predictor, Predictor::Gradient);
        assert_eq!(
            reference
                .modular
                .palette_policy
                .as_ref()
                .unwrap()
                .speed_tier,
            -1
        );
        assert_eq!(
            reference.modular.sampling.as_ref().unwrap().node_threshold,
            61.0
        );
        assert_eq!(
            reference.modular.sampling.as_ref().unwrap().nb_repeats,
            0.5f32 * 0.3f32
        );
        assert_eq!(reference.modular.rct_candidates, Some(19));
        assert_eq!(reference.modular.weighted_candidates, None);
        assert!(atlas.pixels().is_empty());
    }

    #[test]
    fn tectonic_explicit_streaming_keeps_signed_backend_policy() {
        let image = Image::new(1025, 1, PixelLayout::Gray, PixelData::U8(vec![5; 1025])).unwrap();
        let params = CandidateParams {
            speed_tier: -1,
            buffering: SourceBuffering::Streaming,
            group_size_shift: Some(0),
            patches: Override::Off,
            ..Default::default()
        };
        let options = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        assert!(options.modular.palette_policy.as_ref().unwrap().groupwise);
        assert!(options.modular.entropy_options.streaming_mode);
        assert_eq!(
            options.modular.palette_policy.as_ref().unwrap().speed_tier,
            -1
        );
        assert_eq!(
            options.modular.sampling.as_ref().unwrap().node_threshold,
            61.0
        );
        assert_eq!(options.modular.predictor, Predictor::Variable);
        assert!(options.source_patches.is_none());
    }

    #[test]
    fn explicit_tectonic_candidate_reaches_native_encoder_without_effort_aliasing() {
        let image = Image::new(8, 8, PixelLayout::Rgb, PixelData::U8(vec![9; 8 * 8 * 3])).unwrap();
        let params = CandidateParams {
            speed_tier: -1,
            patches: Override::Off,
            ..Default::default()
        };
        let options = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        assert_eq!(
            options
                .modular
                .sampling
                .as_ref()
                .unwrap()
                .max_property_values,
            32
        );
        assert!(!crate::encode(&image, &options).unwrap().is_empty());
        assert_eq!(params.speed_tier, -1);
    }
}

#[cfg(test)]
mod direct_input_tests {
    use super::*;
    use crate::metadata::{BitDepth, ExtraChannelInfo, ExtraChannelType};
    use crate::{PixelData, PixelLayout};

    #[test]
    fn explicit_candidates_accept_typed_depths_without_relaxing_search_gate() {
        let mut images = vec![
            Image::new(
                2,
                1,
                PixelLayout::Gray,
                PixelData::U32(vec![0, 0x7fff_ffff]),
            )
            .unwrap(),
            Image::new(
                2,
                1,
                PixelLayout::Gray,
                PixelData::F16(vec![0x8000, 0x3c00]),
            )
            .unwrap(),
            Image::new(
                2,
                1,
                PixelLayout::Gray,
                PixelData::F32(vec![f32::from_bits(0x7fc1_2345), -0.0]),
            )
            .unwrap(),
        ];
        let custom = Image::new(2, 1, PixelLayout::Gray, PixelData::F32(vec![0.5, -1.0])).unwrap();
        for (bits, exponent) in [(5, 2), (13, 5), (16, 6), (24, 7), (32, 8)] {
            let mut metadata = custom.metadata().clone();
            metadata.bit_depth = BitDepth::floating(bits, exponent).unwrap();
            images.push(custom.clone().with_metadata(metadata, vec![]).unwrap());
        }
        let wide_storage =
            Image::new(2, 1, PixelLayout::Gray, PixelData::U32(vec![0, 65535])).unwrap();
        let mut metadata = wide_storage.metadata().clone();
        metadata.bit_depth = BitDepth::new(16).unwrap();
        images.push(wide_storage.with_metadata(metadata, vec![]).unwrap());
        for image in images {
            let original = image.clone();
            assert!(crate::scheduler::scan::validate_source_input(&image).is_err());
            for speed_tier in [-1, 0, 1, 3, 7, 8] {
                let params = CandidateParams {
                    speed_tier,
                    patches: Override::Off,
                    ..Default::default()
                };
                let options = project_native(&params, &image, &EncodeOptions::default())
                    .unwrap()
                    .exact_options()
                    .unwrap();
                assert_eq!(options.modular.predictor, params.resolved_predictor());
                assert!(options.modular.palette_policy.is_some());
                assert!(options.modular.sampling.is_some());
                assert!(options.source_patches.is_none());
                assert_eq!(image, original, "projection must not convert sample values");
            }
        }
    }

    #[test]
    fn high_color_depth_retains_source_patch_outer_gate_even_when_forced_on() {
        for pixels in [PixelData::U32(vec![0]), PixelData::F32(vec![-0.0])] {
            let image = Image::new(1, 1, PixelLayout::Gray, pixels).unwrap();
            for patches in [Override::Default, Override::On] {
                let params = CandidateParams {
                    patches,
                    ..Default::default()
                };
                let options = project_native(&params, &image, &EncodeOptions::default())
                    .unwrap()
                    .exact_options()
                    .unwrap();
                assert!(options.source_patches.is_none());
                assert!(options.source_patch_reference.is_none());
            }
        }
    }

    #[test]
    fn float_extras_do_not_switch_integer_patch_color_and_atlas_is_metadata_only() {
        let base = Image::new(2, 1, PixelLayout::Rgb, PixelData::U8(vec![9; 6])).unwrap();
        let mut metadata = base.metadata().clone();
        let extras = [
            PixelData::F32(vec![f32::from_bits(0x7fc1_2345), -0.0]),
            PixelData::F16(vec![0x8000, 0x3c00]),
            PixelData::U32(vec![0x7fff_ffff, 0]),
        ];
        for plane in &extras {
            let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
            extra.bit_depth = plane.bit_depth();
            metadata.extra_channels.push(extra);
        }
        let image = base.with_metadata(metadata, extras.to_vec()).unwrap();
        assert!(crate::scheduler::scan::validate_source_input(&image).is_err());
        let params = CandidateParams {
            max_properties: 4,
            ..Default::default()
        };
        let options = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        let patches = options.source_patches.as_ref().unwrap();
        assert!(
            !patches.floating_point,
            "only COLOR controls the detector domain"
        );
        assert_eq!(patches.extra_channel_dim_shifts, [0, 0, 0]);
        let mut planes = vec![vec![0; 8]; 6];
        for color in planes.iter_mut().take(3) {
            color.fill(-1);
        }
        let atlas = image.with_modular_planes(4, 2, planes).unwrap();
        assert!(atlas.pixels().is_empty());
        assert!(atlas.extra_planes().is_empty());
        let resolved = options
            .source_patch_reference
            .as_ref()
            .unwrap()
            .resolve(&atlas)
            .unwrap();
        assert_eq!(resolved.modular.predictor, Predictor::Gradient);
        assert_eq!(
            resolved
                .modular
                .sampling
                .as_ref()
                .unwrap()
                .previous_channels,
            4
        );
        assert_eq!(atlas.metadata(), image.metadata());
    }

    #[test]
    fn float_patch_color_preserves_declared_words_and_resolves_atlas_geometry() {
        let mut images = vec![
            Image::new(
                2,
                1,
                PixelLayout::Gray,
                PixelData::F16(vec![0x8000, 0x3c00]),
            )
            .unwrap(),
        ];
        let custom = Image::new(
            2,
            1,
            PixelLayout::Rgb,
            PixelData::F32(vec![0.5, -0.0, -1.0, -0.5, 1.0, 0.0]),
        )
        .unwrap();
        for (bits, exponent) in [(5, 2), (13, 5), (16, 5), (16, 6)] {
            let mut metadata = custom.metadata().clone();
            metadata.bit_depth = BitDepth::floating(bits, exponent).unwrap();
            metadata
                .extra_channels
                .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 31).unwrap());
            images.push(
                custom
                    .clone()
                    .with_metadata(metadata, vec![PixelData::U32(vec![0, 0x7fff_ffff])])
                    .unwrap(),
            );
        }
        let params = CandidateParams {
            speed_tier: 4,
            group_size_shift: Some(0),
            patches: Override::On,
            predictor: Some(Predictor::Zero),
            max_properties: 3,
            learning_fraction: 0.2,
            ..Default::default()
        };
        for image in images {
            let original = image.clone();
            let mut invalid_planes = vec![vec![0]; image.channel_count()];
            invalid_planes[0][0] = (1u32 << image.bits()) as i32;
            assert!(image.with_modular_planes(1, 1, invalid_planes).is_err());
            assert!(crate::scheduler::scan::validate_source_input(&image).is_err());
            let options = project_native(&params, &image, &EncodeOptions::default())
                .unwrap()
                .exact_options()
                .unwrap();
            let patches = options.source_patches.as_ref().unwrap();
            assert!(patches.floating_point);
            assert_eq!(
                patches.extra_channel_dim_shifts,
                vec![0; image.metadata().extra_channels.len()]
            );
            assert!(
                !options
                    .modular
                    .sampling
                    .as_ref()
                    .unwrap()
                    .properties
                    .contains(&1)
            );
            let recipe = options.source_patch_reference.as_ref().unwrap();
            assert_eq!(recipe.sampling.predictor, Predictor::Gradient);
            assert_eq!(recipe.sampling.learning_fraction, 0.2);
            assert!(recipe.modular.sampling.is_none());
            for (width, height, streaming) in [(4, 2, false), (1025, 1, true)] {
                let count = (width * height) as usize;
                let planes = (0..image.channel_count())
                    .map(|channel| {
                        (0..count)
                            .map(|i| image.modular_sample(channel, i % 2))
                            .collect()
                    })
                    .collect();
                let atlas = image.with_modular_planes(width, height, planes).unwrap();
                assert!(atlas.pixels().is_empty() && atlas.extra_planes().is_empty());
                assert_eq!(atlas.metadata(), image.metadata());
                let reference = recipe.resolve(&atlas).unwrap();
                assert_eq!(reference.group_size_shift, 0);
                let modular = &reference.modular;
                assert_eq!(modular.predictor, Predictor::Gradient);
                assert_eq!(
                    modular.source_lz77.as_ref().unwrap().method,
                    SourceLz77Method::Lz77b1w3t
                );
                assert_eq!(modular.entropy_options.streaming_mode, streaming);
                assert_eq!(
                    modular.palette_policy.as_ref().unwrap().groupwise,
                    streaming
                );
                let sampling = modular.sampling.as_ref().unwrap();
                assert_eq!(sampling.properties.contains(&1), streaming);
                assert_eq!(sampling.max_channel_size, 128);
                assert_eq!(sampling.previous_channels, 3.min(image.channel_count() - 1));
                assert_eq!(sampling.nb_repeats, 0.2f32 * 0.7f32);
                for channel in 0..image.channel_count() {
                    assert_eq!(atlas.channel_depth(channel), image.channel_depth(channel));
                    assert_eq!(
                        atlas.modular_sample(channel, count - 1),
                        image.modular_sample(channel, (count - 1) % 2)
                    );
                }
            }
            assert_eq!(image, original);
        }
    }

    #[test]
    fn subsampled_source_atlas_policy_remains_explicit() {
        let image = Image::new(2, 2, PixelLayout::Gray, PixelData::U8(vec![0; 4])).unwrap();
        let mut metadata = image.metadata().clone();
        let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
        extra.dim_shift = 1;
        metadata.extra_channels.push(extra);
        let image = image
            .with_metadata(metadata, vec![PixelData::U8(vec![0])])
            .unwrap();
        assert!(matches!(
            project_native(&CandidateParams::default(), &image, &EncodeOptions::default()),
            Err(Error::InvalidInput(message)) if message.contains("frozen RoundtripPatchFrame")
        ));
        let params = CandidateParams {
            patches: Override::Off,
            ..Default::default()
        };
        assert!(
            project_native(&params, &image, &EncodeOptions::default())
                .unwrap()
                .exact_options()
                .is_ok()
        );
    }
}

#[cfg(test)]
mod responsive_tests {
    use super::*;
    use crate::modular::{
        encode::ModularOptions,
        group::Channel,
        layout::Passes,
        palette::{PaletteStage, PaletteState, apply_palettes},
        source_options::SourceBuffering,
    };
    use crate::{PixelData, PixelLayout};

    fn image(width: u32, height: u32) -> Image {
        Image::new(
            width,
            height,
            PixelLayout::Rgba,
            PixelData::U8(
                (0..width as usize * height as usize * 4)
                    .map(|i| (i.wrapping_mul(13) ^ (i / 7)) as u8)
                    .collect(),
            ),
        )
        .unwrap()
    }

    #[test]
    fn responsive_selects_global_script_local_trees_and_nonstreaming_entropy() {
        let image = image(1025, 1);
        for speed_tier in -1..=9 {
            for buffering in [SourceBuffering::Auto, SourceBuffering::Streaming] {
                let params = CandidateParams {
                    responsive: true,
                    speed_tier,
                    buffering,
                    group_size_shift: Some(0),
                    patches: Override::Off,
                    ..Default::default()
                };
                let projection =
                    project_native(&params, &image, &EncodeOptions::default()).unwrap();
                let training = projection.training.clone();
                let options = projection.exact_options().unwrap();
                assert!(options.responsive);
                assert!(options.modular.squeeze.is_empty());
                assert_eq!(options.passes, Passes::default());
                assert_eq!(options.shared_tree, speed_tier <= 0);
                assert!(options.global_palette);
                let palette = options.modular.palette_policy.as_ref().unwrap();
                assert!(palette.responsive);
                assert!(!palette.groupwise);
                assert!(!options.modular.entropy_options.streaming_mode);
                assert!(options.modular.source_tree_entropy);
                assert_eq!(options.modular.predictor, Predictor::Zero);
                assert_eq!(options.modular.rct, Some(6));
                assert_eq!(options.modular.rct_candidates, None);
                assert_eq!(options.modular.weighted_candidates, None);
                assert!(options.source_patches.is_none());
                let tier = params.modular_speed_tier();
                assert_eq!(palette.speed_tier, tier);
                assert_eq!(
                    options.modular.fixed_tree,
                    match tier {
                        7 => Some(FixedTree::Weighted),
                        8 => Some(FixedTree::Gradient),
                        _ => None,
                    }
                );
                let sampling = options.modular.sampling.as_ref().unwrap();
                assert_eq!(sampling.nb_repeats, training.fraction);
                assert_eq!(sampling.node_threshold, (75 + 14 * i32::from(tier)) as f32);
                assert_eq!(sampling.max_channel_size, 128);
                assert_eq!(
                    sampling.properties,
                    training
                        .properties
                        .iter()
                        .map(|&p| p as usize)
                        .collect::<Vec<_>>()
                );
                assert_eq!(&sampling.properties[..3], &[0, 1, 4]);
            }
        }
    }

    #[test]
    fn responsive_retains_explicit_predictor_rct_and_disables_palette_stages() {
        let image = image(16, 16);
        for predictor in [Predictor::Zero, Predictor::Gradient, Predictor::Weighted] {
            for (rct, expected) in [(None, Some(6)), (Some(0), None), (Some(17), Some(17))] {
                let params = CandidateParams {
                    responsive: true,
                    speed_tier: 1,
                    predictor: Some(predictor),
                    rct,
                    palette_colors: 70000,
                    delta_palette: DeltaPalette::ExactLeft,
                    patches: Override::Off,
                    ..Default::default()
                };
                let options = project_native(&params, &image, &EncodeOptions::default())
                    .unwrap()
                    .exact_options()
                    .unwrap();
                assert_eq!(options.modular.predictor, predictor);
                assert_eq!(options.modular.rct, expected);
                assert_eq!(options.modular.rct_candidates, None);
                assert_eq!(
                    options.modular.weighted_candidates,
                    (predictor == Predictor::Weighted).then_some(5)
                );
                let palette = options.modular.palette_policy.as_ref().unwrap();
                let original = vec![
                    Channel {
                        samples: vec![0; 16],
                        width: 4,
                        height: 4,
                        hshift: 0,
                        vshift: 0,
                    };
                    3
                ];
                for stage in [PaletteStage::PreRct, PaletteStage::PostRct] {
                    let mut channels = original.clone();
                    let mut state = PaletteState {
                        meta_channels: 0,
                        max_bitdepth: 9,
                        max_value: 255,
                    };
                    assert!(
                        apply_palettes(&mut channels, &mut state, 8, stage, palette)
                            .unwrap()
                            .is_empty()
                    );
                    assert_eq!(state.meta_channels, 0);
                    assert_eq!(channels.len(), 3);
                }
            }
        }
    }

    #[test]
    fn local_directions_never_impersonate_responsive_or_silently_disappear() {
        let image = image(16, 16);
        for responsive in [false, true] {
            for squeeze in [vec![true], vec![true, false, true], vec![true; 31]] {
                let params = CandidateParams {
                    responsive,
                    squeeze,
                    patches: Override::Off,
                    ..Default::default()
                };
                assert_eq!(
                    params.resolved_predictor(),
                    if responsive {
                        Predictor::Zero
                    } else {
                        Predictor::Best
                    }
                );
                let projection =
                    project_native(&params, &image, &EncodeOptions::default()).unwrap();
                assert_eq!(projection.state(), AdaptationState::Unavailable);
                assert!(
                    projection
                        .gaps
                        .contains(&CapabilityGap::ResponsiveTransformOrchestration)
                );
                assert_eq!(
                    projection
                        .gaps
                        .contains(&CapabilityGap::MoreThanThirtySqueezePasses),
                    params.squeeze.len() > 30
                );
                assert!(projection.exact_options().is_err());
                assert!(
                    source_patch_reference_policy(
                        &params,
                        &image,
                        &EncodeOptions::default(),
                        NativeResourcePolicy::default()
                    )
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn default_candidate_does_not_inherit_responsive_local_passes_or_progression() {
        let base = EncodeOptions {
            responsive: true,
            passes: Passes::progressive(),
            modular: ModularOptions {
                squeeze: vec![true, false],
                palette_policy: Some(PaletteOptions {
                    responsive: true,
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        let params = CandidateParams {
            patches: Override::Off,
            ..Default::default()
        };
        let options = project_native(&params, &image(16, 16), &base)
            .unwrap()
            .exact_options()
            .unwrap();
        assert!(!options.responsive);
        assert!(!options.modular.palette_policy.as_ref().unwrap().responsive);
        assert!(options.modular.squeeze.is_empty());
        assert!(options.shared_tree);
        assert_eq!(options.passes, Passes::default());
        assert_eq!(options.modular.predictor, Predictor::Best);
        assert!(base.responsive);
        assert_eq!(base.modular.squeeze, vec![true, false]);
    }

    #[test]
    fn patch_atlas_clears_responsive_before_reselecting_rct_lz_sampling_and_route() {
        let image = image(2304, 1);
        let params = CandidateParams {
            responsive: true,
            speed_tier: 4,
            learning_fraction: 0.2,
            ..Default::default()
        };
        let options = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        assert!(!options.shared_tree);
        assert!(options.source_patches.is_some());
        assert_eq!(
            options.modular.source_lz77.as_ref().unwrap().method,
            SourceLz77Method::Lz77b3w3t
        );
        let recipe = options.source_patch_reference.unwrap();
        assert!(!recipe.sampling.responsive);
        assert_eq!(recipe.sampling.learning_fraction, 0.2);
        for (width, streaming) in [(64, false), (2304, true)] {
            let atlas = image
                .with_modular_planes(width, 1, vec![vec![-1; width as usize]; 4])
                .unwrap();
            assert!(atlas.pixels().is_empty());
            let resolved = recipe.resolve(&atlas).unwrap();
            let modular = resolved.modular;
            assert!(!modular.palette_policy.as_ref().unwrap().responsive);
            assert_eq!(
                modular.palette_policy.as_ref().unwrap().groupwise,
                streaming
            );
            assert_eq!(modular.entropy_options.streaming_mode, streaming);
            assert_eq!(modular.predictor, Predictor::Gradient);
            assert_eq!(modular.rct, None);
            assert_eq!(modular.rct_candidates, Some(5));
            assert_eq!(
                modular.source_lz77.as_ref().unwrap().method,
                SourceLz77Method::Lz77b1w3t
            );
            let sampling = modular.sampling.as_ref().unwrap();
            assert_eq!(sampling.nb_repeats, 0.2f32 * 0.7f32);
            assert_eq!(
                sampling.properties,
                if streaming {
                    vec![0, 1, 15, 9, 10]
                } else {
                    vec![0, 15, 9, 10, 11]
                }
            );
            assert_eq!(atlas.modular_sample(0, 0), -1);
        }
    }

    #[test]
    fn responsive_identity_is_explicit_and_not_an_automatic_search_option() {
        let defaults = CandidateParams::default();
        let responsive = CandidateParams {
            responsive: true,
            ..defaults.clone()
        };
        assert_ne!(defaults, responsive);
        assert!(super::super::policy::default_pixel_options(&defaults));
        assert!(!super::super::policy::default_pixel_options(&responsive));
        assert!(super::super::policy::PixelSearch::new(&image(16, 16), responsive, None).is_err());
    }

    #[test]
    fn responsive_fixed_tiers_reach_the_native_encoder() {
        let image = image(257, 17);
        for speed_tier in [7, 8] {
            let params = CandidateParams {
                responsive: true,
                speed_tier,
                group_size_shift: Some(0),
                patches: Override::Off,
                ..Default::default()
            };
            let options = project_native(&params, &image, &EncodeOptions::default())
                .unwrap()
                .exact_options()
                .unwrap();
            let bytes = crate::encode(&image, &options).unwrap();
            assert!(!bytes.is_empty());
        }
    }
}
