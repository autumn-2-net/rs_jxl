//! Native projection and diagnostic ownership, including bounded encode isolation.
use super::*;
use crate::modular::encode::receipt_cost::ModelCostReceipt;
use crate::{PixelData, PixelLayout, scheduler::params::SourceBuffering};
use std::sync::Arc;

#[test]
fn native_pilot_consensus_requires_adjusted_ranking_not_just_cost_diagnostics() {
    let image = Image::new(16, 16, PixelLayout::Gray, PixelData::U8(vec![7; 256])).unwrap();
    let params = CandidateParams::default().backend(9).unwrap();
    let mut driver = DriverOptions::scout();
    driver.scout_pilot_first = true;
    driver.scout_pilot_consensus = true;
    let mut base = EncodeOptions::default();
    base.modular.model_cost_receipt = Some(Arc::new(ModelCostReceipt::default()));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for acceptance in [
        NativeAcceptance::RequireExact,
        NativeAcceptance::AllowReportedGaps,
    ] {
        let error = run_native_search_with_pool(
            &image,
            &params,
            &driver,
            &base,
            NativeBackendOptions {
                acceptance,
                model_adjusted_probes: false,
                ..Default::default()
            },
            &pool,
            None,
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput(ref message)
            if message == "native scout pilot consensus requires model-adjusted probe ranking"));
        assert_eq!(base.modular.model_cost_receipt.as_ref().unwrap().bits(), 0);
    }
}

#[test]
fn native_representation_receipts_are_opt_in_and_private_to_each_preparation() {
    let image = Image::new(16, 8, PixelLayout::Gray, PixelData::U8(vec![7; 128])).unwrap();
    let params = CandidateParams::default().backend(9).unwrap();
    let settings = NativeBackendOptions {
        record_representations: true,
        ..Default::default()
    };
    let base = EncodeOptions::default();
    let (first, first_record) =
        prepare(&params, &image, &base, settings, NativeTrialKind::Full).unwrap();
    let (_, second) = prepare(&params, &image, &base, settings, NativeTrialKind::Probe).unwrap();
    let a = first_record.representation.unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &a,
        first
            .options
            .modular
            .representation_receipt
            .as_ref()
            .unwrap()
    ));
    assert!(!std::sync::Arc::ptr_eq(&a, &second.representation.unwrap()));
    assert!(a.get().is_none());
    let (plain, record) = prepare(
        &params,
        &image,
        &base,
        Default::default(),
        NativeTrialKind::Full,
    )
    .unwrap();
    assert!(plain.options.modular.representation_receipt.is_none());
    assert!(record.representation.is_none());
}

#[test]
fn diagnostic_palette_forcing_is_probe_only_and_reported_not_source_exact() {
    let image = Image::new(16, 8, PixelLayout::Gray, PixelData::U8(vec![7; 128])).unwrap();
    let params = crate::scheduler::policy::literal(&CandidateParams::default(), 9, 0.01).unwrap();
    let settings = NativeBackendOptions {
        probe_palette: true,
        ..Default::default()
    };
    let base = EncodeOptions::default();
    assert!(prepare(&params, &image, &base, settings, NativeTrialKind::Probe).is_err());
    let (full, full_record) =
        prepare(&params, &image, &base, settings, NativeTrialKind::Full).unwrap();
    assert!(full_record.gaps.is_empty());
    assert!(!full.options.modular.palette_policy.unwrap().force_palette);
    let settings = NativeBackendOptions {
        acceptance: NativeAcceptance::AllowReportedGaps,
        ..settings
    };
    let (probe, record) =
        prepare(&params, &image, &base, settings, NativeTrialKind::Probe).unwrap();
    assert!(probe.options.modular.palette_policy.unwrap().force_palette);
    assert_eq!(record.gaps, vec![CapabilityGap::ForcedProbePalette]);
    for predictor in [None, Some(crate::modular::predictor::Predictor::Variable)] {
        let p = CandidateParams {
            predictor,
            ..params.clone()
        };
        let (probe, record) = prepare(&p, &image, &base, settings, NativeTrialKind::Probe).unwrap();
        assert!(!probe.options.modular.palette_policy.unwrap().force_palette);
        assert!(record.gaps.is_empty());
    }
}

#[test]
fn scout_enables_exact_channel_gather_without_overwriting_explicit_gather() {
    let base = EncodeOptions {
        threads: 8,
        ..Default::default()
    };
    let ordinary = scout_execution_options(&base, false);
    assert!(ordinary.modular.source_gather.is_none());
    assert!(!ordinary.modular.parallel_tokens);
    assert!(ordinary.modular.source_lz77.is_none());
    let tuned = scout_execution_options(&base, true);
    assert!(tuned.modular.parallel_tokens);
    let gather = tuned.modular.source_gather.as_ref().unwrap();
    assert_eq!(gather.workers, 8);
    assert_eq!(gather.stripes, 1);
    assert!(gather.sparse);
    let lz = tuned.modular.source_lz77.as_ref().unwrap();
    assert!(lz.compact_steps);
    let workers = lz.parallelism.unwrap();
    assert_eq!(workers.max_workers, 8);
    assert!(workers.fast_streams);
    assert!(base.modular.source_gather.is_none());
    let mut explicit = base;
    explicit.modular.source_gather = Some(crate::modular::sample::GatherOptions {
        workers: 2,
        sparse: false,
        ..Default::default()
    });
    explicit.modular.source_lz77 = Some(crate::tools::lz77::SourceConfig {
        parallelism: Some(crate::tools::lz77::SourceParallelism {
            max_workers: 2,
            fast_streams: false,
            ..Default::default()
        }),
        ..Default::default()
    });
    let preserved = scout_execution_options(&explicit, true);
    let gather = preserved.modular.source_gather.unwrap();
    assert_eq!(gather.workers, 2);
    assert!(!gather.sparse);
    let lz = preserved.modular.source_lz77.unwrap();
    assert!(!lz.compact_steps);
    assert_eq!(lz.parallelism.unwrap().max_workers, 2);
    assert!(!lz.parallelism.unwrap().fast_streams);
}

#[test]
fn probe_model_estimation_is_explicit_and_uses_fresh_receipts() {
    let image = Image::new(16, 16, PixelLayout::Gray, PixelData::U8(vec![0; 256])).unwrap();
    let base = EncodeOptions::default();
    let params = CandidateParams::default().backend(9).unwrap();
    let strict = NativeBackendOptions {
        model_adjusted_probes: true,
        ..Default::default()
    };
    assert!(prepare(&params, &image, &base, strict, NativeTrialKind::Probe).is_err());
    let settings = NativeBackendOptions {
        acceptance: NativeAcceptance::AllowReportedGaps,
        ..strict
    };
    let (a, record) = prepare(&params, &image, &base, settings, NativeTrialKind::Probe).unwrap();
    let (b, _) = prepare(&params, &image, &base, settings, NativeTrialKind::Probe).unwrap();
    assert!(record.gaps.contains(&CapabilityGap::ModelScaledProbeScore));
    let costs = record.model_costs.as_ref().unwrap();
    assert!(Arc::ptr_eq(
        costs,
        a.options.modular.model_cost_receipt.as_ref().unwrap()
    ));
    let reference = a.options.source_patch_reference.as_ref().unwrap();
    assert!(Arc::ptr_eq(
        costs,
        reference.modular.model_cost_receipt.as_ref().unwrap()
    ));
    assert!(Arc::ptr_eq(
        costs,
        reference
            .resolve(&image)
            .unwrap()
            .modular
            .model_cost_receipt
            .as_ref()
            .unwrap()
    ));
    assert!(!std::sync::Arc::ptr_eq(
        a.options.modular.model_cost_receipt.as_ref().unwrap(),
        b.options.modular.model_cost_receipt.as_ref().unwrap()
    ));
    let (full, record) = prepare(&params, &image, &base, strict, NativeTrialKind::Full).unwrap();
    assert!(full.options.modular.model_cost_receipt.is_none());
    assert!(record.model_costs.is_none());
    assert!(record.gaps.is_empty());
}

#[test]
fn base_model_receipts_request_fresh_full_and_probe_diagnostics() {
    let image = Image::new(16, 16, PixelLayout::Gray, PixelData::U8(vec![7; 256])).unwrap();
    let params = CandidateParams::default().backend(9).unwrap();
    let mut base = EncodeOptions::default();
    let inherited = Arc::new(ModelCostReceipt::default());
    base.modular.model_cost_receipt = Some(Arc::clone(&inherited));
    let mut receipts = vec![Arc::clone(&inherited)];
    for adjusted in [false, true] {
        for kind in [NativeTrialKind::Full, NativeTrialKind::Probe] {
            let settings = NativeBackendOptions {
                acceptance: if adjusted && kind == NativeTrialKind::Probe {
                    NativeAcceptance::AllowReportedGaps
                } else {
                    NativeAcceptance::RequireExact
                },
                model_adjusted_probes: adjusted,
                record_representations: true,
                ..Default::default()
            };
            let (artifact, record) = prepare(&params, &image, &base, settings, kind).unwrap();
            let costs = record.model_costs.unwrap();
            assert_eq!(costs.bits(), 0);
            assert!(receipts.iter().all(|old| !Arc::ptr_eq(old, &costs)));
            assert!(Arc::ptr_eq(
                &costs,
                artifact
                    .options
                    .modular
                    .model_cost_receipt
                    .as_ref()
                    .unwrap()
            ));
            let reference = artifact.options.source_patch_reference.as_ref().unwrap();
            assert!(Arc::ptr_eq(
                &costs,
                reference.modular.model_cost_receipt.as_ref().unwrap()
            ));
            assert!(record.representation.is_some());
            assert!(reference.modular.representation_receipt.is_none());
            assert_eq!(
                record.gaps.contains(&CapabilityGap::ModelScaledProbeScore),
                adjusted && kind == NativeTrialKind::Probe
            );
            receipts.push(costs);
        }
    }
    assert_eq!(inherited.bits(), 0);
    assert!(Arc::ptr_eq(
        &inherited,
        base.modular.model_cost_receipt.as_ref().unwrap()
    ));

    // A diagnostic already installed in a base reference policy is also an
    // opt-in request, never a collector to reuse after candidate projection.
    let (artifact, _) = prepare(
        &params,
        &image,
        &base,
        Default::default(),
        NativeTrialKind::Full,
    )
    .unwrap();
    let reference_costs = artifact.options.modular.model_cost_receipt.clone().unwrap();
    base.modular.model_cost_receipt = None;
    base.source_patch_reference = artifact.options.source_patch_reference;
    let (artifact, record) = prepare(
        &params,
        &image,
        &base,
        Default::default(),
        NativeTrialKind::Full,
    )
    .unwrap();
    let costs = record.model_costs.unwrap();
    assert!(!Arc::ptr_eq(&costs, &reference_costs));
    assert!(Arc::ptr_eq(
        &costs,
        artifact
            .options
            .modular
            .model_cost_receipt
            .as_ref()
            .unwrap()
    ));
    assert!(record.gaps.is_empty());
}

#[test]
fn bounded_model_costs_do_not_leak_into_later_full_or_probe_invocations() {
    let image = Image::new(
        257,
        3,
        PixelLayout::Gray,
        PixelData::U8((0..257 * 3).map(|i| (i % 251) as u8).collect()),
    )
    .unwrap();
    let params = CandidateParams {
        speed_tier: 9,
        group_size_shift: Some(0),
        patches: Override::Off,
        palette_colors: 0,
        channel_colors_percent: 0.0,
        channel_colors_pre_transform_percent: 0.0,
        ..Default::default()
    };
    let mut base = EncodeOptions::default();
    let inherited = Arc::new(ModelCostReceipt::default());
    base.modular.model_cost_receipt = Some(Arc::clone(&inherited));
    let (bounded, record) = prepare(
        &params,
        &image,
        &base,
        Default::default(),
        NativeTrialKind::Full,
    )
    .unwrap();
    let old = record.model_costs.unwrap();
    assert!(matches!(
        encode_full_trial(&image, &bounded.options, Some(1), None, None).unwrap(),
        TrialEncode::Bounded { lower_bound: 1 }
    ));
    let discarded_bits = old.bits();
    assert!(discarded_bits > 0);
    let mut completed = Vec::new();
    for kind in [NativeTrialKind::Full, NativeTrialKind::Probe] {
        let (artifact, record) = prepare(&params, &image, &base, Default::default(), kind).unwrap();
        let costs = record.model_costs.unwrap();
        assert!(!Arc::ptr_eq(&old, &costs));
        assert!(completed.iter().all(|prior| !Arc::ptr_eq(prior, &costs)));
        assert_eq!(costs.bits(), 0);
        let bytes = crate::encode::encode_serial(&image, &artifact.options).unwrap();
        assert!(costs.bits() > 0 && costs.bits() <= bytes.len() * 8);
        assert_eq!(old.bits(), discarded_bits);
        assert_eq!(inherited.bits(), 0);
        completed.push(costs);
    }
}

#[test]
fn explicit_token_parallel_option_survives_full_probe_and_reference_preparation() {
    let image = Image::new(1025, 1, PixelLayout::Gray, PixelData::U8(vec![1; 1025])).unwrap();
    assert!(!EncodeOptions::default().modular.parallel_tokens);
    for enabled in [false, true] {
        let mut base = EncodeOptions::default();
        base.modular.parallel_tokens = enabled;
        for buffering in [SourceBuffering::Auto, SourceBuffering::FullFrame] {
            let params = CandidateParams {
                speed_tier: 1,
                group_size_shift: Some(0),
                buffering,
                ..Default::default()
            };
            for kind in [NativeTrialKind::Full, NativeTrialKind::Probe] {
                let (artifact, record) = prepare(
                    &params,
                    &image,
                    &base,
                    NativeBackendOptions::default(),
                    kind,
                )
                .unwrap();
                assert!(record.gaps.is_empty());
                assert_eq!(artifact.options.modular.parallel_tokens, enabled);
                let streaming = buffering == SourceBuffering::Auto;
                assert_eq!(
                    artifact
                        .options
                        .modular
                        .palette_policy
                        .as_ref()
                        .unwrap()
                        .groupwise,
                    streaming
                );
                assert_eq!(
                    artifact.options.modular.entropy_options.streaming_mode,
                    streaming
                );
                if streaming {
                    assert!(artifact.options.source_patch_reference.is_none());
                } else {
                    let policy = artifact
                        .options
                        .source_patch_reference
                        .as_ref()
                        .expect("full-frame source patch gate");
                    assert_eq!(policy.modular.parallel_tokens, enabled);
                    assert_eq!(
                        policy.resolve(&image).unwrap().modular.parallel_tokens,
                        enabled
                    );
                }
            }
        }
    }
}
