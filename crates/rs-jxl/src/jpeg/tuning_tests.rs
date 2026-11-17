use super::*;

#[test]
fn ac_audit_is_explicit_and_accounts_for_selected_model() {
    assert!(
        TranscodeOptions::default()
            .resolve()
            .unwrap()
            .ac_audit
            .is_none()
    );
    for name in ["gray_seq_17x9", "rgb_seq_33x25"] {
        let input = super::super::transcode_reference_tests::fixtures::fixture(name).unwrap();
        for (effort, threads, ac_lz) in [
            (3, 1, AcLzSearch::Off),
            (9, 4, AcLzSearch::Source),
            (10, 1, AcLzSearch::FastForced),
        ] {
            let options = TranscodeOptions {
                effort,
                threads,
                ..Default::default()
            };
            let tuning = Options {
                ac_lz,
                ..Default::default()
            };
            let ordinary = transcode(&input, &options, tuning).unwrap();
            let (output, sizes, audit) = transcode_audited(&input, &options, tuning).unwrap();
            assert_eq!(
                output.len(),
                ordinary.len(),
                "diagnostic must not alter decisions"
            );
            assert_eq!(
                crate::decode::reconstruct_jpeg(&output, &Default::default()).unwrap(),
                input
            );
            assert_eq!(sizes.total_bits(), output.len() * 8);
            let a = audit.entropy;
            assert!(a.tokens > 0);
            assert_eq!(sizes.ac_lz_enabled, Some(a.lz77));
            assert!(a.raw_context_entropy_bits <= a.cluster_entropy_bits + 1e-6);
            assert!(a.cluster_entropy_bits <= a.model_symbol_bits + 1e-6);
            assert!(a.extra_bit_entropy_bits <= a.extra_bits as f64 + 1e-6);
            assert!(a.extra_bits as usize + audit.selector_bits <= sizes.ac_data_bits);
            assert!(audit.seconds.is_finite());
        }
    }
}

#[test]
fn context_maps_and_orders_reconstruct_original_bytes() {
    for name in ["gray_seq_17x9", "rgb_seq_33x25"] {
        let original = super::super::transcode_reference_tests::fixtures::fixture(name).unwrap();
        for block_context in [
            BlockContextPolicy::Source,
            BlockContextPolicy::ChannelOnly,
            BlockContextPolicy::SharedChroma,
            BlockContextPolicy::PairedLuma,
            BlockContextPolicy::Quantile4,
            BlockContextPolicy::Quantile5,
            BlockContextPolicy::Activity4,
            BlockContextPolicy::Activity5,
        ] {
            for ac_order in [
                AcOrderPolicy::Source,
                AcOrderPolicy::Zigzag,
                AcOrderPolicy::EdgeFirst,
                AcOrderPolicy::EdgeLast,
                AcOrderPolicy::BandEdgeFirst,
                AcOrderPolicy::BandEdgeLast,
            ] {
                let (encoded, costs) = transcode_measured(
                    &original,
                    &TranscodeOptions {
                        effort: 9,
                        threads: 1,
                        ..Default::default()
                    },
                    Options {
                        block_context,
                        ac_order,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_eq!(costs.total_bits(), encoded.len() * 8);
                assert!((1..=16).contains(&costs.ac_block_contexts.unwrap()));
                assert_eq!(
                    crate::decode::reconstruct_jpeg(&encoded, &Default::default()).unwrap(),
                    original,
                    "{name} {block_context:?} {ac_order:?}"
                );
            }
        }
    }
}

#[test]
fn legacy_pairing_cannot_silently_override_new_context_policy() {
    let original =
        super::super::transcode_reference_tests::fixtures::fixture("rgb_seq_33x25").unwrap();
    let error = transcode(
        &original,
        &Default::default(),
        Options {
            paired_luma: true,
            block_context: BlockContextPolicy::ChannelOnly,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("paired_luma cannot be combined"));
    assert!(matches!(error, crate::Error::InvalidInput(_)));
}

#[test]
fn explicit_policies_preserve_resource_limit_errors() {
    use super::super::{
        TranscodeLimits, coefficient_reader::CoefficientReadLimits, reader::ReadLimits,
    };
    use crate::Error;

    let original =
        super::super::transcode_reference_tests::fixtures::fixture("rgb_seq_33x25").unwrap();
    let tuning = Options {
        cfl_fit: CflFitPolicy::ActualMaximum,
        symmetric_ma: true,
        paired_luma: true,
        ..Default::default()
    };
    for (name, limits) in [
        (
            "input bytes",
            TranscodeLimits {
                input: CoefficientReadLimits {
                    structural: ReadLimits {
                        max_input_bytes: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "input coefficient storage",
            TranscodeLimits {
                input: CoefficientReadLimits {
                    max_coefficient_bytes: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "encoder coefficient storage",
            TranscodeLimits {
                max_coefficients: 1,
                ..Default::default()
            },
        ),
        (
            "tokens",
            TranscodeLimits {
                max_tokens: 1,
                ..Default::default()
            },
        ),
        (
            "MA selected samples",
            TranscodeLimits {
                sample_limit: 1,
                ..Default::default()
            },
        ),
        (
            "output bytes",
            TranscodeLimits {
                max_output_bytes: 1,
                ..Default::default()
            },
        ),
    ] {
        let options = TranscodeOptions {
            effort: 9,
            threads: 4,
            cooperative_ma: Some(Default::default()),
            limits,
            ..Default::default()
        };
        let error = transcode(&original, &options, tuning).unwrap_err();
        assert!(matches!(error, Error::Limit(_)), "{name}: {error:?}");
    }
}

#[test]
fn explicit_policies_preserve_original_jpeg_in_serial_and_parallel() {
    for name in ["gray_seq_17x9", "rgb_seq_33x25"] {
        let original = super::super::transcode_reference_tests::fixtures::fixture(name).unwrap();
        for tuning in [
            Options::default(),
            Options {
                cfl_fit: CflFitPolicy::ActualMaximum,
                ..Default::default()
            },
            Options {
                symmetric_ma: true,
                ..Default::default()
            },
            Options {
                paired_luma: true,
                ..Default::default()
            },
            Options {
                cfl_fit: CflFitPolicy::ActualMaximum,
                symmetric_ma: true,
                paired_luma: true,
                ..Default::default()
            },
        ] {
            for threads in [1, 4] {
                let options = TranscodeOptions {
                    effort: 9,
                    threads,
                    cooperative_ma: Some(Default::default()),
                    ..Default::default()
                };
                let (encoded, sizes) = transcode_measured(&original, &options, tuning).unwrap();
                assert_eq!(sizes.total_bits(), encoded.len() * 8);
                let restored =
                    crate::decode::reconstruct_jpeg(&encoded, &Default::default()).unwrap();
                assert_eq!(restored, original, "{name} {tuning:?} t{threads}");
            }
        }
    }
}

#[test]
fn explicit_ac_parsers_reconstruct_original_bytes() {
    for name in ["gray_seq_17x9", "rgb_seq_33x25"] {
        let original = super::super::transcode_reference_tests::fixtures::fixture(name).unwrap();
        for ac_lz in [
            AcLzSearch::Off,
            AcLzSearch::Fast,
            AcLzSearch::Optimal,
            AcLzSearch::FastForced,
            AcLzSearch::OptimalForced,
        ] {
            for threads in [1, 4] {
                let options = TranscodeOptions {
                    effort: 9,
                    threads,
                    ..Default::default()
                };
                let (encoded, costs) = transcode_measured(
                    &original,
                    &options,
                    Options {
                        ac_lz,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_eq!(costs.total_bits(), encoded.len() * 8);
                if matches!(ac_lz, AcLzSearch::FastForced | AcLzSearch::OptimalForced) {
                    assert_eq!(costs.ac_lz_enabled, Some(true));
                }
                assert_eq!(
                    crate::decode::reconstruct_jpeg(&encoded, &Default::default()).unwrap(),
                    original
                );
            }
        }
    }
}
