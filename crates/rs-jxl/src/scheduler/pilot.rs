//! Experimental content-aware pilot policy. Probe estimates choose work, not
//! output bounds; the winner is always a completed, retained native encoding.
use super::{
    CandidateParams, DriverOptions, NativeAcceptance, NativeBackendOptions, NativeResourcePolicy,
    NativeSearchResult, run_native_pixel_search,
};
use crate::tools::lz77::{SourceConfig, SourceFastPrefix, SourceFastRanges};
use crate::{EncodeOptions, Error, Image, Result};

/// Resource ceilings and per-invocation diagnostics for the opt-in pilot preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PilotOptions {
    /// Source MA whole-tree ceiling; None uses the decoder-derived bound,
    /// not base's direct encoder preset. Some(n) remains a hard caller cap.
    pub resources: NativeResourcePolicy,
    /// Allocate an independent representation receipt for each native trial.
    pub record_representations: bool,
    /// Experimental: score every family on fine regions without a coarse pass.
    /// Retains fine screening and full winner validation; default is false.
    pub single_stage: bool,
    /// Experimental: run independent serial probe roots with the same scratch cap.
    pub independent_probes: bool,
    /// Overlap a bounded initial wave of small full-frame trials; on in this preset.
    pub parallel_initial: bool,
    /// Experimental: wake cooperative workers only when their group completes.
    pub quiet_arena: bool,
    /// Overlap a reusable automatic full trial with pilot analysis when the
    /// content/resource gates permit it; on in this preset.
    pub overlap_automatic: bool,
    /// Experimental: quarter the MA sampling fraction in pilot probes only.
    pub probe_sampling_quarter: bool,
    /// Continuous-tone near-tie consensus, independent of model-adjusted ranking;
    /// on in this preset. Probe estimates are not final-size guarantees.
    pub near_tie_consensus: bool,
}

impl Default for PilotOptions {
    fn default() -> Self {
        Self {
            resources: NativeResourcePolicy::default(),
            record_representations: false,
            single_stage: false,
            independent_probes: false,
            parallel_initial: true,
            quiet_arena: false,
            overlap_automatic: true,
            probe_sampling_quarter: false,
            near_tie_consensus: true,
        }
    }
}

fn policy(
    image: &Image,
    literal: bool,
    threads: usize,
    options: PilotOptions,
) -> (DriverOptions, NativeBackendOptions) {
    let mut driver = DriverOptions::scout();
    driver.scout_pilot_first = true;
    driver.scout_small_full = true;
    driver.scout_native_rct_aliases = image.channel_count() < 3;
    driver.scout_parallel_finalists = true;
    driver.scout_initial_wave = options.parallel_initial;
    driver.scout_overlap_automatic = options.overlap_automatic;
    driver.workers.tasks = threads.clamp(1, 64);
    driver.workers.scratch_bytes = 256 << 20;
    driver.cooperative_probes = !options.independent_probes;
    driver.arena_quiet = options.quiet_arena;
    driver.longest_first_probes = true;
    driver.preflight.enabled = true;
    driver.screen.adaptive_coverage = true;
    driver.scout_pilot_consensus = !literal;
    driver.scout_near_tie_consensus = options.near_tie_consensus;
    driver.scout_representation_witness = true;
    driver.plan.extended_rct = literal && image.color_channels() == 3;
    driver.scout_staged_probes = !literal && !options.single_stage;
    driver.scout_coarse_consensus = !literal && !options.single_stage;
    driver.screen.regions.full_width_strips = true;
    driver.probe_workers = Some(if options.independent_probes {
        driver.workers.concurrency(
            threads.clamp(1, 64),
            super::worker_pool::PROBE_SCRATCH_BYTES,
        )
    } else {
        threads.clamp(1, 64)
    });
    let native = NativeBackendOptions {
        acceptance: NativeAcceptance::AllowReportedGaps,
        probe_palette: literal,
        literal_palette_tools: literal,
        model_adjusted_probes: !literal,
        probe_sampling_quarter: options.probe_sampling_quarter,
        routed_tools: true,
        resources: options.resources,
        record_representations: options.record_representations,
        ..Default::default()
    };
    (driver, native)
}

fn execution_options(base: &EncodeOptions, literal: bool) -> EncodeOptions {
    let mut options = base.clone();
    if !literal && options.modular.source_gather.is_none() {
        options.modular.source_gather = Some(crate::modular::sample::GatherOptions {
            workers: base.threads.clamp(1, 64),
            sparse: true,
            concat_batches: true,
            ..Default::default()
        });
    }
    if !options.modular.source_ma_parallel.subtrees {
        options.modular.source_ma_parallel.workers.tasks = base.threads.clamp(1, 64);
    }
    options
        .modular
        .source_lz77
        .get_or_insert_with(|| SourceConfig {
            compact_steps: true,
            fast_ranges: SourceFastRanges::Both,
            fast_prefix: Some(SourceFastPrefix::Simd),
            ..Default::default()
        });
    options
}

/// Opt-in e9/e10 pilot search. This is not the frozen C++ search policy and is
/// not a size bound against exhaustive search. All adaptations are reported.
/// Uses default [`PilotOptions`]; see [`encode_with_options`] for tool defaults
/// and explicit resource/diagnostic controls.
pub fn encode(
    image: &Image,
    params: &CandidateParams,
    base: &EncodeOptions,
) -> Result<NativeSearchResult> {
    encode_with_options(image, params, base, PilotOptions::default())
}

/// Opt-in pilot search with explicit source resource ceilings and diagnostics.
/// Pilot coverage is available from 2^17 pixels; ordinary scout keeps its gate.
/// Screens retain unadjusted strip scores; non-literal inputs amortize emitted
/// tree/model costs and require regional agreement to skip close alternatives.
///
/// `base` retains thread, pixel/output and MA sample limits. Source MA's leaf
/// ceiling comes from `options.resources`, NOT `base.modular.max_tree_leaves`;
/// its default follows the source decoder's whole-tree bound; an explicit cap
/// remains a hard limit. Exhaustion errors by default; the explicit modular
/// MA budget mode can stop optional splits losslessly at that limit.
/// Source candidate projection owns predictor/transform/tree policy. The routed
/// preset enables weighted/subtree parallelism and family-specific MA/LZ tools.
/// With no explicit subtree strategy, MA tasks follow threads (at most 64);
/// an enabled subtree strategy retains its worker settings. Missing source LZ
/// configuration gets compact steps, Both fast ranges and a SIMD prefix filter.
/// Existing source LZ configuration is retained before family-specific routing;
/// missing gather/LZ parallelism is filled by native scout execution defaults.
/// Probe slots follow threads (at most 64). No process-wide defaults are changed.
/// Non-literal inputs with no explicit gather configuration use bounded terminal
/// sample concatenation. Explicit gather settings and ordinary source are retained.
/// With fewer than three total channels, otherwise identical cross-channel
/// RCT0/6 candidates share the RCT0 trial; learned/static tools remain distinct.
pub fn encode_with_options(
    image: &Image,
    params: &CandidateParams,
    base: &EncodeOptions,
    options: PilotOptions,
) -> Result<NativeSearchResult> {
    if u64::from(image.width()) * u64::from(image.height()) > base.max_pixels {
        return Err(Error::Limit("pixel count"));
    }
    if !matches!(params.speed_tier, 0 | 1) {
        return Err(Error::InvalidInput(
            "pilot search requires effort 9 or 10".into(),
        ));
    }
    super::scan::validate_source_input(image)?;
    let literal = super::scan::literal_evidence(image);
    let (driver, native) = policy(image, literal, base.threads, options);
    let base = execution_options(base, literal);
    run_native_pixel_search(image, params, &driver, &base, native)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::native::{NativeTrialKind, prepare};
    use crate::{PixelData, PixelLayout};
    use std::sync::Arc;

    #[test]
    fn palette_tool_defaults_are_content_gated_and_explicit_policy_is_retained() {
        use crate::modular::palette::{PaletteOptions, PaletteOrder};
        use crate::modular::predictor::Predictor;
        let mut base = EncodeOptions::default();
        let image = image();
        for literal in [false, true] {
            let (_, native) = policy(&image, literal, 1, PilotOptions::default());
            assert!(
                execution_options(&base, literal)
                    .modular
                    .palette_policy
                    .is_none()
            );
            for predictor in [None, Some(Predictor::Zero), Some(Predictor::Variable)] {
                let params = CandidateParams {
                    predictor,
                    ..CandidateParams::default().backend(9).unwrap()
                };
                let (artifact, _) =
                    prepare(&params, &image, &base, native, NativeTrialKind::Full).unwrap();
                let palette = artifact.options.modular.palette_policy.unwrap();
                let enabled = literal && predictor == Some(Predictor::Zero);
                assert_eq!(palette.force_palette, enabled);
                assert_eq!(
                    palette.tools.order == PaletteOrder::FrequencyBandsMorton,
                    enabled
                );
                assert!(!palette.zero_tokens);
            }
        }
        let explicit = PaletteOptions {
            force_palette: false,
            ..Default::default()
        };
        base.modular.palette_policy = Some(explicit.clone());
        assert_eq!(
            execution_options(&base, true).modular.palette_policy,
            Some(explicit)
        );
        let (_, native) = policy(&image, true, 1, PilotOptions::default());
        let params = CandidateParams {
            predictor: Some(Predictor::Zero),
            ..CandidateParams::default().backend(9).unwrap()
        };
        let (artifact, _) = prepare(&params, &image, &base, native, NativeTrialKind::Full).unwrap();
        let palette = artifact.options.modular.palette_policy.unwrap();
        assert!(!palette.force_palette);
        assert_ne!(palette.tools.order, PaletteOrder::FrequencyBandsMorton);
    }

    #[test]
    fn preset_defaults_and_explicit_near_tie_overrides_are_forwarded() {
        let preset = PilotOptions::default();
        assert!(preset.parallel_initial && preset.overlap_automatic && preset.near_tie_consensus);
        assert!(!preset.probe_sampling_quarter && !preset.single_stage);
        for literal in [false, true] {
            for enabled in [false, true] {
                let (driver, native) = policy(
                    &image(),
                    literal,
                    16,
                    PilotOptions {
                        near_tie_consensus: enabled,
                        ..Default::default()
                    },
                );
                assert_eq!(driver.scout_near_tie_consensus, enabled);
                assert_eq!(driver.scout_pilot_consensus, !literal);
                assert_eq!(native.model_adjusted_probes, !literal);
            }
        }
    }

    #[test]
    fn concat_default_is_natural_pilot_only_and_retains_explicit_gather() {
        let mut base = EncodeOptions {
            threads: 16,
            ..Default::default()
        };
        let natural = execution_options(&base, false);
        let gather = natural.modular.source_gather.as_ref().unwrap();
        assert!(gather.concat_batches && gather.sparse);
        assert!(!gather.cache_hashes);
        assert_eq!(gather.workers, 16);
        assert!(
            execution_options(&base, true)
                .modular
                .source_gather
                .is_none()
        );
        assert!(base.modular.source_gather.is_none());
        base.modular.source_gather = Some(crate::modular::sample::GatherOptions {
            workers: 2,
            ..Default::default()
        });
        for literal in [true, false] {
            let tuned = execution_options(&base, literal);
            let gather = tuned.modular.source_gather.unwrap();
            assert_eq!(gather.workers, 2);
            assert!(!gather.concat_batches && !gather.sparse);
        }
    }

    #[test]
    fn independent_probes_keep_worker_and_scratch_estimate_caps() {
        for threads in [1, 4, 8, 16, 64, 256] {
            let (driver, _) = policy(
                &image(),
                false,
                threads,
                PilotOptions {
                    independent_probes: true,
                    ..Default::default()
                },
            );
            assert!(!driver.cooperative_probes);
            assert_eq!(driver.probe_workers, Some(threads.min(8)));
            assert_eq!(driver.workers.tasks, threads.min(64));
            assert!(driver.scout_parallel_finalists);
            assert!(driver.scout_initial_wave);
        }
        let (driver, _) = policy(
            &image(),
            false,
            16,
            PilotOptions {
                parallel_initial: true,
                ..Default::default()
            },
        );
        assert!(driver.scout_initial_wave && driver.cooperative_probes);
    }

    #[test]
    fn single_stage_skips_only_the_coarse_pass() {
        for literal in [false, true] {
            let (driver, native) = policy(
                &image(),
                literal,
                8,
                PilotOptions {
                    single_stage: true,
                    ..Default::default()
                },
            );
            assert!(!driver.scout_staged_probes);
            assert!(!driver.scout_coarse_consensus);
            assert!(driver.scout_pilot_first && driver.screen.regions.full_width_strips);
            assert_eq!(native.model_adjusted_probes, !literal);
            assert_eq!(native.probe_palette, literal);
        }
        let result = encode_with_options(
            &image(),
            &CandidateParams::default().backend(9).unwrap(),
            &EncodeOptions {
                threads: 1,
                ..Default::default()
            },
            PilotOptions {
                single_stage: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!result.output.bytes.is_empty());
    }

    fn image() -> Image {
        Image::new(16, 8, PixelLayout::Gray, PixelData::U8(vec![7; 128])).unwrap()
    }

    #[test]
    fn native_rct_alias_policy_counts_extra_channels_not_just_color_channels() {
        use crate::metadata::{ExtraChannelInfo, ExtraChannelType};

        let gray = image();
        let mut metadata = gray.metadata().clone();
        metadata.extra_channels =
            vec![ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap(); 2];
        let extras = gray
            .clone()
            .with_metadata(metadata, vec![PixelData::U8(vec![0; 128]); 2])
            .unwrap();
        for (image, aliases) in [
            (gray, true),
            (
                Image::new(16, 8, PixelLayout::GrayAlpha, PixelData::U8(vec![7; 256])).unwrap(),
                true,
            ),
            (
                Image::new(16, 8, PixelLayout::Rgb, PixelData::U8(vec![7; 384])).unwrap(),
                false,
            ),
            (extras, false),
        ] {
            for literal in [false, true] {
                let (driver, _) = policy(&image, literal, 8, PilotOptions::default());
                assert!(driver.scout_pilot_first);
                assert_eq!(driver.scout_native_rct_aliases, aliases);
            }
        }
    }

    #[test]
    fn screenshot_scores_are_not_model_amortized() {
        let (driver, native) = policy(&image(), true, 8, PilotOptions::default());
        assert!(driver.scout_pilot_first && driver.screen.regions.full_width_strips);
        assert!(native.probe_palette && native.routed_tools);
        assert!(!native.model_adjusted_probes);
        assert!(!driver.scout_pilot_consensus && !driver.scout_staged_probes);
        assert!(!driver.scout_coarse_consensus);
        assert!(!native.record_representations);
    }

    #[test]
    fn natural_confidence_uses_cost_corrected_scores() {
        let (driver, native) = policy(&image(), false, 8, PilotOptions::default());
        assert!(driver.scout_pilot_first && driver.screen.regions.full_width_strips);
        assert!(driver.scout_pilot_consensus && driver.scout_staged_probes);
        assert!(native.model_adjusted_probes && native.routed_tools);
        assert!(!native.probe_palette);
        assert!(driver.scout_coarse_consensus);
    }

    #[test]
    fn smaller_pilot_gate_does_not_change_ordinary_scout() {
        for height in [255, 256, 512, 768, 1024] {
            let image = Image::new(
                512,
                height,
                PixelLayout::Gray,
                PixelData::U8(vec![7; 512 * height as usize]),
            )
            .unwrap();
            let pixels = 512 * u64::from(height);
            for literal in [false, true] {
                let (driver, _) = policy(&image, literal, 8, PilotOptions::default());
                assert!(driver.preflight.enabled && driver.screen.adaptive_coverage);
                let screen =
                    super::super::CandidateScreen::new(&image, None, driver.screen, |_, _| Ok(1))
                        .unwrap();
                assert_eq!(screen.available(), pixels >= 1 << 17);
            }
            let ordinary = DriverOptions::scout();
            assert!(!ordinary.preflight.enabled && !ordinary.screen.adaptive_coverage);
            let screen =
                super::super::CandidateScreen::new(&image, None, ordinary.screen, |_, _| Ok(1))
                    .unwrap();
            assert_eq!(screen.available(), pixels >= 1 << 19);
        }
    }

    #[test]
    fn default_execution_workers_follow_threads_without_mutating_base() {
        for threads in [1, 8, 16, 64, 256] {
            let base = EncodeOptions {
                threads,
                ..Default::default()
            };
            let original_tasks = base.modular.source_ma_parallel.workers.tasks;
            let tuned = execution_options(&base, false);
            assert_eq!(
                tuned.modular.source_ma_parallel.workers.tasks,
                threads.min(64)
            );
            assert!(!tuned.modular.source_ma_parallel.subtrees);
            let lz = tuned.modular.source_lz77.as_ref().unwrap();
            assert!(lz.compact_steps);
            assert_eq!(lz.fast_ranges, SourceFastRanges::Both);
            assert_eq!(lz.fast_prefix, Some(SourceFastPrefix::Simd));
            assert!(lz.parallelism.is_none());
            assert!(base.modular.source_lz77.is_none());
            assert_eq!(
                base.modular.source_ma_parallel.workers.tasks,
                original_tasks
            );
            for literal in [false, true] {
                let (driver, _) = policy(&image(), literal, threads, PilotOptions::default());
                assert_eq!(driver.probe_workers, Some(threads.min(64)));
            }
        }
    }

    #[test]
    fn explicit_execution_settings_survive_helper_defaults() {
        let mut base = EncodeOptions {
            threads: 16,
            ..Default::default()
        };
        base.modular.source_ma_parallel.subtrees = true;
        base.modular.source_ma_parallel.workers.tasks = 3;
        base.modular.source_ma_parallel.quiet = true;
        base.modular.source_lz77 = Some(SourceConfig {
            fast_ranges: SourceFastRanges::RepeatedInsert,
            fast_prefix: Some(SourceFastPrefix::Disabled),
            parallelism: Some(crate::tools::lz77::SourceParallelism {
                max_workers: 2,
                fast_streams: false,
                ..Default::default()
            }),
            ..Default::default()
        });
        let tuned = execution_options(&base, false);
        assert!(tuned.modular.source_ma_parallel.subtrees);
        assert_eq!(tuned.modular.source_ma_parallel.workers.tasks, 3);
        assert!(tuned.modular.source_ma_parallel.quiet);
        let lz = tuned.modular.source_lz77.as_ref().unwrap();
        assert!(!lz.compact_steps);
        assert_eq!(lz.fast_ranges, SourceFastRanges::RepeatedInsert);
        assert_eq!(lz.fast_prefix, Some(SourceFastPrefix::Disabled));
        assert_eq!(lz.parallelism.unwrap().max_workers, 2);
        assert!(!lz.parallelism.unwrap().fast_streams);
        base.modular.source_lz77 = Some(SourceConfig::default());
        let explicit_defaults = execution_options(&base, false);
        let lz = explicit_defaults.modular.source_lz77.unwrap();
        assert!(!lz.compact_steps);
        assert_eq!(lz.fast_ranges, SourceFastRanges::Disabled);
        assert_eq!(lz.fast_prefix, None);
    }

    #[test]
    fn resource_and_receipt_options_reach_each_native_preparation() {
        let image = image();
        let params = CandidateParams::default().backend(9).unwrap();
        let mut base = EncodeOptions {
            threads: 4,
            max_pixels: 128,
            max_output_bytes: 8192,
            ..Default::default()
        };
        base.modular.max_tree_leaves = 1;
        base.modular.sample_limit = 17;
        let mut receipts = Vec::new();
        for literal in [false, true] {
            for leaves in [None, Some(7)] {
                for record_representations in [false, true] {
                    let options = PilotOptions {
                        resources: NativeResourcePolicy {
                            max_tree_leaves: leaves,
                        },
                        record_representations,
                        single_stage: false,
                        independent_probes: false,
                        parallel_initial: false,
                        quiet_arena: false,
                        overlap_automatic: false,
                        probe_sampling_quarter: false,
                        near_tie_consensus: false,
                    };
                    let (_, settings) = policy(&image, literal, base.threads, options);
                    assert_eq!(settings.resources, options.resources);
                    for kind in [NativeTrialKind::Full, NativeTrialKind::Probe] {
                        let (artifact, record) = prepare(
                            &params,
                            &image,
                            &execution_options(&base, literal),
                            settings,
                            kind,
                        )
                        .unwrap();
                        let expected_leaves =
                            leaves.unwrap_or(crate::modular::ma_train::MAX_LEAVES);
                        assert_eq!(record.resources.max_tree_leaves, expected_leaves);
                        assert_eq!(artifact.options.modular.max_tree_leaves, expected_leaves);
                        assert_eq!(record.resources.sample_limit, 17);
                        assert_eq!(artifact.options.modular.sample_limit, 17);
                        assert_eq!(artifact.options.max_pixels, base.max_pixels);
                        assert_eq!(artifact.options.max_output_bytes, base.max_output_bytes);
                        assert_eq!(artifact.options.threads, base.threads);
                        assert_eq!(record.representation.is_some(), record_representations);
                        if let Some(receipt) = record.representation {
                            assert!(Arc::ptr_eq(
                                &receipt,
                                artifact
                                    .options
                                    .modular
                                    .representation_receipt
                                    .as_ref()
                                    .unwrap()
                            ));
                            assert!(
                                receipts
                                    .iter()
                                    .all(|previous| !Arc::ptr_eq(previous, &receipt))
                            );
                            receipts.push(receipt);
                        }
                    }
                }
            }
        }
        assert_eq!(base.modular.max_tree_leaves, 1);
        assert_eq!(base.modular.sample_limit, 17);
        assert!(base.modular.source_lz77.is_none());
        assert!(base.modular.representation_receipt.is_none());
    }

    #[test]
    fn pixel_limit_is_checked_before_other_options_and_native_dispatch() {
        let image = image();
        let mut base = EncodeOptions {
            threads: 0,
            max_pixels: 127,
            ..Default::default()
        };
        let params = CandidateParams::default();
        assert!(matches!(
            encode(&image, &params, &base),
            Err(Error::Limit("pixel count"))
        ));
        assert!(matches!(
            encode_with_options(&image, &params, &base, PilotOptions::default()),
            Err(Error::Limit("pixel count"))
        ));
        base.max_pixels = 128;
        assert!(
            matches!(encode(&image, &params, &base), Err(Error::InvalidInput(message))
            if message == "pilot search requires effort 9 or 10")
        );
        let params = params.backend(9).unwrap();
        assert!(
            matches!(encode_with_options(&image, &params, &base, PilotOptions::default()),
            Err(Error::InvalidInput(message)) if message == "threads must be 1..=256")
        );
    }
}
