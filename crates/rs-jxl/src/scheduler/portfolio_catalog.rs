//! Native, single-candidate portfolio recipes; no search or feedback state.
//!
//! Baseline is exactly the ordinary source projection at the requested effort
//! (e9 is tier 1), not patch-off or a lower-effort substitute. Each recipe is
//! rebuilt from `base`; never pass a previous candidate's options as `base`.
//! Callers must keep base execution/resource settings fixed across a race and
//! validate the image with the normal encoder. Metadata eligibility here does
//! not promise tool activation, a dictionary, or a compression gain.
//! Process-global profiling/experiment environment must be scrubbed by the
//! caller; this catalog neither reads nor mutates process environment.

use super::{
    adapter,
    params::{CandidateParams, DeltaPalette, Override, SourceBuffering},
};
use crate::modular::{
    fast_context,
    ma_train::{
        DescriptionCost, LookaheadOptions, SeedMode, SeedOptions, SeedTopology, TreeLearning,
    },
    palette::{PaletteOptions, PaletteOrder},
    palette_groups::Grouping,
    refinement::{ContextMapOptions, OffsetMode, OffsetOptions},
};
use crate::tools::patches::{RepresentativeMode, TilePatchOptions};
use crate::{EncodeOptions, Error, Image, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    Baseline,
    Offset,
    CrossChannel,
    Shallow,
    Learning,
    Entropy,
    PaletteOrder,
    Delta,
    Grouping,
    Source,
    Tile,
}

macro_rules! catalog {
    ($($variant:ident => ($name:literal, $family:ident)),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Candidate { $($variant),+ }

        pub const ALL: &[Candidate] = &[$(Candidate::$variant),+];

        impl Candidate {
            pub const fn name(self) -> &'static str {
                match self { $(Self::$variant => $name),+ }
            }

            pub const fn family(self) -> Family {
                match self { $(Self::$variant => Family::$family),+ }
            }
        }
    };
}

catalog! {
    Baseline => ("baseline", Baseline),
    OffsetDistribution => ("offset-distribution", Offset),
    Prev1 => ("prev1", CrossChannel),
    Prev2 => ("prev2", CrossChannel),
    Prev1Distribution => ("prev1-distribution", CrossChannel),
    Prev2Distribution => ("prev2-distribution", CrossChannel),
    ShallowFull => ("shallow-full", Shallow),
    ShallowFullPrev1 => ("shallow-full-prev1", Shallow),
    ShallowFullDistribution => ("shallow-full-distribution", Shallow),
    Activity81 => ("activity81", Shallow),
    Activity25Refine => ("activity25-refine", Shallow),
    Activity81Refine => ("activity81-refine", Shallow),
    LearningCost => ("learning-cost", Learning),
    LearningSymmetric => ("learning-symmetric", Learning),
    DescriptionHalf => ("description-half", Learning),
    Lookahead2 => ("lookahead2", Learning),
    ContextMap => ("context-map", Entropy),
    PaletteRunFrequency => ("palette-run-frequency", PaletteOrder),
    PaletteHot128 => ("palette-hot128", PaletteOrder),
    PaletteMinLa => ("palette-minla", PaletteOrder),
    PaletteEzeng => ("palette-ezeng", PaletteOrder),
    PaletteEzengMinLa => ("palette-ezeng-minla", PaletteOrder),
    DeltaGradient => ("delta-gradient", Delta),
    DeltaLeft => ("delta-left", Delta),
    DeltaTop => ("delta-top", Delta),
    ZeroGradient => ("zero-gradient", Delta),
    ZeroLeft => ("zero-left", Delta),
    ZeroTop => ("zero-top", Delta),
    PaletteRg => ("palette-rg", Grouping),
    PaletteYCo => ("palette-yco", Grouping),
    PaletteYCg => ("palette-ycg", Grouping),
    PaletteCoCg => ("palette-cocg", Grouping),
    PaletteYaCoCg => ("palette-ya-cocg", Grouping),
    Source => ("source", Source),
    SourceParallel => ("source-parallel", Source),
    SourcePrev2 => ("source-prev2", Source),
    SourceDistribution => ("source-distribution", Source),
    SourceMedoid => ("source-medoid", Source),
    SourceMinEntropy => ("source-min-entropy", Source),
    Tile16 => ("tile16", Tile),
    TileOffGrid => ("tile-off-grid", Tile),
}

impl Candidate {
    fn offsets(self) -> bool {
        matches!(
            self,
            Self::OffsetDistribution
                | Self::Prev1Distribution
                | Self::Prev2Distribution
                | Self::ShallowFullDistribution
                | Self::SourceDistribution
        )
    }

    fn previous_channels(self) -> u8 {
        match self {
            Self::Prev1 | Self::Prev1Distribution | Self::ShallowFullPrev1 => 1,
            Self::Prev2 | Self::Prev2Distribution | Self::SourcePrev2 => 2,
            _ => 0,
        }
    }

    /// Returns a structural exclusion reason, not a heuristic admission score.
    /// A supported recipe may still decline its transform/detector at encode time.
    pub fn supports(self, image: &Image, effort: u8) -> Option<&'static str> {
        if !(1..=11).contains(&effort) {
            return Some("source effort must be 1..=11");
        }
        if effort <= 3
            && (self.offsets() && self != Self::ShallowFullDistribution
                || matches!(self.family(), Family::Learning | Family::CrossChannel))
        {
            return Some("candidate requires a learned source tree (effort >= 4)");
        }
        if self.previous_channels() != 0 && image.channel_count() < 2 {
            return Some("cross-channel candidate requires at least two channels");
        }
        if self.family() == Family::Grouping {
            let metadata = image.metadata();
            if image.color_channels() != 3
                || metadata.bit_depth.is_float()
                || metadata
                    .extra_channels
                    .iter()
                    .any(|ec| ec.dim_shift != 0 || ec.bit_depth != metadata.bit_depth)
            {
                return Some(
                    "partial palettes require full-resolution equal-depth integer RGB/extras",
                );
            }
            if self == Self::PaletteYaCoCg
                && !metadata
                    .extra_channels
                    .iter()
                    .any(|ec| ec.channel_type == crate::metadata::ExtraChannelType::Alpha)
            {
                return Some("YA/CoCg palette requires an Alpha extra channel");
            }
        }
        if self.family() == Family::Tile
            && (image.metadata().bit_depth.is_float() || image.bits() > 16)
        {
            return Some("exact tile discovery requires integer main channels of at most 16 bits");
        }
        if self.family() == Family::Source {
            if effort < 5 || image.bits() > 16 {
                return Some("Source Add requires effort >= 5 and main depth <= 16");
            }
            if image
                .metadata()
                .extra_channels
                .iter()
                .any(|ec| ec.dim_shift > 3)
            {
                return Some("Source Add does not support extra-channel shifts above 3");
            }
        }
        None
    }

    /// Constructs one native recipe without encoding or running a tournament.
    /// SourcePrev2 applies prev2 to both the main frame and deferred atlas;
    /// SourceDistribution likewise refines both domains. Source means On, not
    /// Off: native depth/effort/seed gates remain authoritative. PaletteEzeng is
    /// the insertion seed alone (zero MinLA rounds), not EZeng + refinement.
    /// PaletteEzengMinLa adds four refinement rounds. SourceParallel alone adds
    /// patch_lab's Channels sparse gather to main and atlas domains, with at
    /// most 64 workers; one-thread calls keep serial gathering. Other Source
    /// variants do not implicitly enable parallel gather or Waves.
    pub fn build(self, image: &Image, base: &EncodeOptions, effort: u8) -> Result<EncodeOptions> {
        validate_base(base)?;
        if let Some(reason) = self.supports(image, effort) {
            return Err(Error::InvalidInput(format!("{}: {reason}", self.name())));
        }
        let mut params = CandidateParams::default().backend(effort)?;
        let mut template = base.clone();
        params.max_properties = self.previous_channels();
        if self.family() == Family::Source {
            params.patches = Override::On;
        }
        if self.family() == Family::Tile {
            params.patches = Override::Off;
        }
        if self.offsets() {
            template.modular.refinement.offsets = Some(OffsetOptions {
                mode: OffsetMode::Distribution,
                ..Default::default()
            });
        }
        match self {
            Self::ShallowFull | Self::ShallowFullPrev1 | Self::ShallowFullDistribution => {
                template.modular.fast_context = Some(fast_context::Options {
                    mode: fast_context::Mode::Auto12,
                    bias: fast_context::Bias::Static,
                    predictors: fast_context::PredictorSet::Full,
                    cross_channel: self == Self::ShallowFullPrev1,
                    ..Default::default()
                });
            }
            Self::Activity81 => {
                template.modular.fast_context = Some(fast_context::Options {
                    mode: fast_context::Mode::Activity81,
                    ..Default::default()
                });
            }
            Self::Activity25Refine | Self::Activity81Refine => {
                template.modular.source_ma_seed = Some(SeedOptions {
                    mode: SeedMode::Refine,
                    topology: if self == Self::Activity25Refine {
                        SeedTopology::Activity25
                    } else {
                        SeedTopology::Activity81
                    },
                    ..Default::default()
                });
            }
            Self::LearningCost => template.modular.tree_learning = TreeLearning::Cost,
            Self::LearningSymmetric => template.modular.tree_learning = TreeLearning::Symmetric,
            Self::DescriptionHalf => {
                template.modular.source_ma_scoring.description = DescriptionCost::Half;
            }
            Self::Lookahead2 => {
                template.modular.source_ma_lookahead = Some(LookaheadOptions {
                    width: 2,
                    ..Default::default()
                });
            }
            Self::ContextMap => {
                template.modular.refinement.context_map = Some(ContextMapOptions {
                    shortlist: 8,
                    max_merges: 4,
                });
            }
            Self::PaletteRunFrequency
            | Self::PaletteHot128
            | Self::PaletteMinLa
            | Self::PaletteEzeng
            | Self::PaletteEzengMinLa => {
                let mut policy = PaletteOptions::default();
                policy.tools.order = match self {
                    Self::PaletteRunFrequency => PaletteOrder::RunFrequency,
                    Self::PaletteHot128 => PaletteOrder::HotRuns { hot_colors: 128 },
                    Self::PaletteMinLa => PaletteOrder::AdjacencyMinLa { rounds: 4 },
                    Self::PaletteEzengMinLa => PaletteOrder::EzengMinLa { rounds: 4 },
                    _ => PaletteOrder::EzengMinLa { rounds: 0 },
                };
                template.modular.palette_policy = Some(policy);
            }
            Self::DeltaGradient => params.delta_palette = DeltaPalette::ExactGradient,
            Self::DeltaLeft => params.delta_palette = DeltaPalette::ExactLeft,
            Self::DeltaTop => params.delta_palette = DeltaPalette::ExactTop,
            Self::ZeroGradient => params.delta_palette = DeltaPalette::ZeroGradient,
            Self::ZeroLeft => params.delta_palette = DeltaPalette::ZeroLeft,
            Self::ZeroTop => params.delta_palette = DeltaPalette::ZeroTop,
            Self::PaletteRg
            | Self::PaletteYCo
            | Self::PaletteYCg
            | Self::PaletteCoCg
            | Self::PaletteYaCoCg => {
                params.buffering = SourceBuffering::FullFrame;
                params.palette_colors = 4096;
                template.modular.palette_policy = Some(PaletteOptions {
                    grouping: Some(match self {
                        Self::PaletteRg => Grouping::Rg,
                        Self::PaletteYCo => Grouping::YCo,
                        Self::PaletteYCg => Grouping::YCg,
                        Self::PaletteCoCg => Grouping::CoCg,
                        _ => Grouping::YaCoCg,
                    }),
                    ..Default::default()
                });
            }
            _ => {}
        }
        // These native tools explicitly reject preparation sharing. Other
        // recipes retain it: immutable preparation keys include palette policy,
        // and learned trees / sample sets are never cached here.
        if template.modular.fast_context.is_some()
            || template.modular.source_ma_seed.is_some()
            || template.modular.source_ma_lookahead.is_some()
        {
            template.modular.candidate_cache = None;
        }
        let mut options = adapter::project_native(&params, image, &template)?.exact_options()?;
        if self == Self::SourceParallel && options.threads > 1 {
            let gather = crate::modular::sample::GatherOptions {
                mode: crate::modular::sample::GatherMode::Channels,
                workers: options.threads.min(64),
                sparse: true,
                ..Default::default()
            };
            options.modular.source_gather = Some(gather.clone());
            if let Some(reference) = options.source_patch_reference.as_mut() {
                reference.modular.source_gather = Some(gather);
            }
        }
        if self.family() == Family::Source {
            let config = options.source_patches.as_mut().ok_or_else(|| {
                Error::InvalidInput("Source Add is disabled by the native frame policy".into())
            })?;
            config.representative.mode = match self {
                Self::SourceMedoid => RepresentativeMode::Medoid,
                Self::SourceMinEntropy => RepresentativeMode::MinEntropy,
                _ => RepresentativeMode::First,
            };
        }
        if self.family() == Family::Tile {
            options.patches = None;
            options.source_patches = None;
            options.source_patch_reference = None;
            options.tile_patches = Some(TilePatchOptions {
                tile_size: 16,
                off_grid: self == Self::TileOffGrid,
                force: false,
                ..Default::default()
            });
        }
        options.modular.validate()?;
        Ok(options)
    }
}

fn validate_base(base: &EncodeOptions) -> Result<()> {
    let m = &base.modular;
    // Receipts are invocation-owned; sharing one across baseline and trials is
    // misleading even when it cannot change the bitstream. Counterfactual replay
    // is especially unsafe: projection otherwise drops its hook silently.
    let palette_interference = m.palette_policy.as_ref().is_some_and(|p| {
        p.counterfactual.is_some()
            || p.receipt.is_some()
            || p.force_palette
            || p.zero_tokens
            || p.direct
            || p.grouping.is_some()
            || p.mode != crate::modular::palette::PaletteMode::Absolute
            || p.tools != Default::default()
    });
    if palette_interference
        || m.fast_context.is_some()
        || m.block_context.is_some()
        || m.source_ma_seed.is_some()
        || m.source_ma_lookahead.is_some()
        || m.tree_learning != TreeLearning::Legacy
        || m.source_ma_scoring != Default::default()
        || m.source_ma_search != Default::default()
        || m.source_channel_tree != Default::default()
        || m.refinement.entropy
        || m.refinement.leaves
        || m.refinement.offsets.is_some()
        || m.refinement.context_map.is_some()
        || m.refinement.receipt.is_some()
        || m.tree_learning_receipt.is_some()
        || m.palette_group_receipt.is_some()
        || m.representation_receipt.is_some()
        || m.model_cost_receipt.is_some()
        || m.custom_tree.is_some()
        || m.fixed_tree.is_some()
        || m.tree_groups.is_some()
        || !m.tree_group_splits.is_empty()
        || base.patches.is_some()
        || base.tile_patches.is_some()
        || base.source_patches.is_some()
        || base.source_patch_reference.is_some()
    {
        return Err(Error::InvalidInput(
            "portfolio base must not contain experimental policies, debug receipts or patch overrides".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout};

    fn image() -> Image {
        Image::new(8, 8, PixelLayout::Rgb, PixelData::U8(vec![0; 8 * 8 * 3])).unwrap()
    }

    #[test]
    fn catalog_names_are_unique_and_stable() {
        let mut names = std::collections::HashSet::new();
        for &candidate in ALL {
            assert!(names.insert(candidate.name()));
            assert!(
                candidate
                    .name()
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
            );
        }
        assert_eq!(ALL.len(), 41);
        assert_eq!(ALL[0], Candidate::Baseline);
        assert_eq!(Candidate::SourceMinEntropy.name(), "source-min-entropy");
    }

    #[test]
    fn baseline_is_unmodified_projection_and_builds_do_not_leak() {
        let image = Image::new(8, 8, PixelLayout::Rgba, PixelData::U8(vec![0; 8 * 8 * 4])).unwrap();
        let base = EncodeOptions::default();
        let expected = adapter::project_native(
            &CandidateParams::default().backend(9).unwrap(),
            &image,
            &base,
        )
        .unwrap()
        .exact_options()
        .unwrap();
        let before = format!("{base:?}");
        for &candidate in ALL {
            candidate.build(&image, &base, 9).unwrap();
        }
        let actual = Candidate::Baseline.build(&image, &base, 9).unwrap();
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        assert_eq!(format!("{base:?}"), before);
        assert_eq!(
            actual.modular.refinement.offsets.unwrap().mode,
            OffsetMode::DistributionSinglePass
        );
        assert_eq!(
            actual.modular.source_ma_scoring.description,
            DescriptionCost::Half
        );
        assert_eq!(actual.modular.previous_channels, 0);
    }

    #[test]
    fn offsets_cross_channel_and_shallow_mapping() {
        let image = image();
        let base = EncodeOptions::default();
        let options = Candidate::Prev2Distribution
            .build(&image, &base, 9)
            .unwrap();
        assert_eq!(options.modular.previous_channels, 2);
        assert_eq!(
            options.modular.sampling.as_ref().unwrap().previous_channels,
            2
        );
        assert_eq!(
            options.modular.refinement.offsets.unwrap().mode,
            OffsetMode::Distribution
        );
        let options = Candidate::ShallowFullPrev1.build(&image, &base, 9).unwrap();
        let fast = options.modular.fast_context.unwrap();
        assert_eq!(fast.mode, fast_context::Mode::Auto12);
        assert_eq!(fast.predictors, fast_context::PredictorSet::Full);
        assert_eq!(fast.bias, fast_context::Bias::Static);
        assert!(fast.cross_channel);
        assert_eq!(options.modular.previous_channels, 1);
    }

    #[test]
    fn learning_entropy_and_palette_mappings() {
        let image = image();
        let base = EncodeOptions::default();
        let build = |candidate: Candidate| candidate.build(&image, &base, 9).unwrap().modular;
        assert_eq!(
            build(Candidate::LearningCost).tree_learning,
            TreeLearning::Cost
        );
        assert_eq!(
            build(Candidate::LearningSymmetric).tree_learning,
            TreeLearning::Symmetric
        );
        assert_eq!(
            build(Candidate::DescriptionHalf)
                .source_ma_scoring
                .description,
            DescriptionCost::Half
        );
        assert_eq!(
            build(Candidate::Lookahead2)
                .source_ma_lookahead
                .unwrap()
                .width,
            2
        );
        assert_eq!(
            build(Candidate::Activity25Refine)
                .source_ma_seed
                .unwrap()
                .topology,
            SeedTopology::Activity25
        );
        assert_eq!(
            build(Candidate::Activity81Refine)
                .source_ma_seed
                .unwrap()
                .mode,
            SeedMode::Refine
        );
        assert_eq!(
            build(Candidate::Activity81).fast_context.unwrap().mode,
            fast_context::Mode::Activity81
        );
        let map = build(Candidate::ContextMap).refinement.context_map.unwrap();
        assert_eq!((map.shortlist, map.max_merges), (8, 4));
        for (candidate, expected) in [
            (Candidate::PaletteRunFrequency, PaletteOrder::RunFrequency),
            (
                Candidate::PaletteHot128,
                PaletteOrder::HotRuns { hot_colors: 128 },
            ),
            (
                Candidate::PaletteMinLa,
                PaletteOrder::AdjacencyMinLa { rounds: 4 },
            ),
            (
                Candidate::PaletteEzeng,
                PaletteOrder::EzengMinLa { rounds: 0 },
            ),
            (
                Candidate::PaletteEzengMinLa,
                PaletteOrder::EzengMinLa { rounds: 4 },
            ),
        ] {
            assert_eq!(
                build(candidate).palette_policy.unwrap().tools.order,
                expected
            );
        }
        for (candidate, expected) in [
            (Candidate::DeltaGradient, DeltaPalette::ExactGradient),
            (Candidate::DeltaLeft, DeltaPalette::ExactLeft),
            (Candidate::DeltaTop, DeltaPalette::ExactTop),
            (Candidate::ZeroGradient, DeltaPalette::ZeroGradient),
            (Candidate::ZeroLeft, DeltaPalette::ZeroLeft),
            (Candidate::ZeroTop, DeltaPalette::ZeroTop),
        ] {
            let params = CandidateParams {
                delta_palette: expected,
                ..CandidateParams::default().backend(9).unwrap()
            };
            let expected = adapter::project_native(&params, &image, &base)
                .unwrap()
                .exact_options()
                .unwrap();
            assert_eq!(
                build(candidate).palette_policy.unwrap().mode,
                expected.modular.palette_policy.unwrap().mode
            );
        }
    }

    #[test]
    fn grouping_source_and_tile_mappings() {
        let image = image();
        let base = EncodeOptions::default();
        for (candidate, expected) in [
            (Candidate::PaletteRg, Grouping::Rg),
            (Candidate::PaletteYCo, Grouping::YCo),
            (Candidate::PaletteYCg, Grouping::YCg),
            (Candidate::PaletteCoCg, Grouping::CoCg),
        ] {
            let options = candidate.build(&image, &base, 9).unwrap();
            let palette = options.modular.palette_policy.unwrap();
            assert_eq!(palette.grouping, Some(expected));
            assert_eq!(palette.palette_colors, 4096);
            assert!(!palette.groupwise);
        }
        for (candidate, expected) in [
            (Candidate::Source, RepresentativeMode::First),
            (Candidate::SourceMedoid, RepresentativeMode::Medoid),
            (Candidate::SourceMinEntropy, RepresentativeMode::MinEntropy),
        ] {
            let options = candidate.build(&image, &base, 9).unwrap();
            assert_eq!(
                options.source_patches.unwrap().representative.mode,
                expected
            );
        }
        let source = Candidate::SourcePrev2.build(&image, &base, 9).unwrap();
        assert_eq!(source.modular.previous_channels, 2);
        assert_eq!(
            source
                .source_patch_reference
                .unwrap()
                .sampling
                .max_previous_channels,
            2
        );
        let source = Candidate::SourceDistribution
            .build(&image, &base, 9)
            .unwrap();
        assert!(source.modular.refinement.offsets.is_some());
        assert!(
            source
                .source_patch_reference
                .unwrap()
                .modular
                .refinement
                .offsets
                .is_some()
        );
        for candidate in [Candidate::Tile16, Candidate::TileOffGrid] {
            let options = candidate.build(&image, &base, 9).unwrap();
            assert!(options.patches.is_none() && options.source_patches.is_none());
            assert!(options.source_patch_reference.is_none());
            let tile = options.tile_patches.unwrap();
            assert!(!tile.force);
            assert_eq!(tile.tile_size, 16);
            assert_eq!(tile.off_grid, candidate == Candidate::TileOffGrid);
        }
    }

    #[test]
    fn source_parallel_gather_is_explicit_bounded_and_targets_both_domains() {
        let image = image();
        for threads in [1, 8, 128] {
            let base = EncodeOptions {
                threads,
                ..Default::default()
            };
            let options = Candidate::SourceParallel.build(&image, &base, 9).unwrap();
            let reference = options.source_patch_reference.as_ref().unwrap();
            if threads == 1 {
                assert!(options.modular.source_gather.is_none());
                assert!(reference.modular.source_gather.is_none());
            } else {
                for gather in [
                    options.modular.source_gather.as_ref().unwrap(),
                    reference.modular.source_gather.as_ref().unwrap(),
                ] {
                    assert!(matches!(
                        gather.mode,
                        crate::modular::sample::GatherMode::Channels
                    ));
                    assert_eq!(gather.workers, threads.min(64));
                    assert!(gather.sparse);
                }
            }
            for candidate in [
                Candidate::Baseline,
                Candidate::Source,
                Candidate::SourcePrev2,
                Candidate::SourceDistribution,
            ] {
                let options = candidate.build(&image, &base, 9).unwrap();
                assert!(options.modular.source_gather.is_none());
                if let Some(reference) = options.source_patch_reference {
                    assert!(reference.modular.source_gather.is_none());
                }
            }
            assert!(base.modular.source_gather.is_none());
        }
    }

    #[test]
    fn rejects_interference_and_structural_mismatches() {
        let image = image();
        let mut base = EncodeOptions::default();
        base.modular.refinement.leaves = true;
        assert!(Candidate::Baseline.build(&image, &base, 9).is_err());
        base = EncodeOptions::default();
        base.modular.palette_policy = Some(PaletteOptions {
            force_palette: true,
            ..Default::default()
        });
        assert!(Candidate::Prev1.build(&image, &base, 9).is_err());
        base = EncodeOptions::default();
        assert!(
            Candidate::OffsetDistribution
                .build(&image, &base, 3)
                .is_err()
        );
        assert!(Candidate::LearningCost.build(&image, &base, 3).is_err());
        assert!(Candidate::Source.build(&image, &base, 4).is_err());
        assert!(Candidate::PaletteYaCoCg.build(&image, &base, 9).is_err());
        assert!(Candidate::Baseline.build(&image, &base, 0).is_err());
        let gray = Image::new(8, 8, PixelLayout::Gray, PixelData::U8(vec![0; 64])).unwrap();
        assert!(Candidate::Prev1.build(&gray, &base, 9).is_err());
        assert!(Candidate::PaletteRg.build(&gray, &base, 9).is_err());
    }

    #[test]
    fn clears_only_native_incompatible_preparation_caches() {
        let image = image();
        let scope = crate::modular::candidate_cache::CandidateCache::scope(&image, 1024 * 1024);
        let mut base = EncodeOptions::default();
        base.modular.candidate_cache = Some(scope.cache.clone());
        for candidate in [
            Candidate::Baseline,
            Candidate::Prev1,
            Candidate::OffsetDistribution,
            Candidate::PaletteHot128,
            Candidate::DeltaLeft,
        ] {
            let options = candidate.build(&image, &base, 9).unwrap();
            assert!(std::sync::Arc::ptr_eq(
                options.modular.candidate_cache.as_ref().unwrap(),
                &scope.cache
            ));
        }
        for candidate in [
            Candidate::ShallowFull,
            Candidate::Activity81,
            Candidate::Activity25Refine,
            Candidate::Activity81Refine,
            Candidate::Lookahead2,
        ] {
            let options = candidate.build(&image, &base, 9).unwrap();
            assert!(options.modular.candidate_cache.is_none());
            if let Some(reference) = options.source_patch_reference {
                assert!(reference.modular.candidate_cache.is_none());
            }
        }
        assert!(base.modular.candidate_cache.is_some());
    }
}
