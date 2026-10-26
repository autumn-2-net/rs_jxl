use super::*;
use crate::{PixelData, PixelLayout};

#[test]
fn quarter_sampling_is_probe_only_reported_and_never_a_resource_relaxation() {
    let image = Image::new(
        32,
        16,
        PixelLayout::Rgb,
        PixelData::U8(vec![7; 32 * 16 * 3]),
    )
    .unwrap();
    let mut base = EncodeOptions::default();
    base.modular.sample_limit = 123;
    let settings = NativeBackendOptions {
        acceptance: NativeAcceptance::AllowReportedGaps,
        probe_sampling_quarter: true,
        ..Default::default()
    };
    for fraction in [0.0, 0.001, 0.01, 0.1, 0.5, 1.0] {
        let params = CandidateParams {
            learning_fraction: fraction,
            ..CandidateParams::default().backend(9).unwrap()
        };
        for kind in [NativeTrialKind::Full, NativeTrialKind::Probe] {
            let (ordinary, ordinary_record) = prepare(
                &params,
                &image,
                &base,
                NativeBackendOptions {
                    probe_sampling_quarter: false,
                    ..settings
                },
                kind,
            )
            .unwrap();
            let (artifact, record) = prepare(&params, &image, &base, settings, kind).unwrap();
            let changed = kind == NativeTrialKind::Probe && fraction >= 0.01;
            assert_eq!(record.requested, params);
            assert_eq!(
                record.effective.learning_fraction,
                fraction * if changed { 0.25 } else { 1.0 }
            );
            assert_eq!(
                record
                    .gaps
                    .contains(&CapabilityGap::ReducedProbeTrainingSamples),
                changed
            );
            assert_eq!(artifact.options.modular.sample_limit, 123);
            assert_eq!(
                artifact.options.modular.max_tree_leaves,
                ordinary.options.modular.max_tree_leaves
            );
            assert_eq!(
                artifact.options.group_size_shift,
                ordinary.options.group_size_shift
            );
            assert_eq!(
                artifact.options.modular.predictor,
                ordinary.options.modular.predictor
            );
            if !changed {
                assert_eq!(record.effective, ordinary_record.effective);
                assert_eq!(record.gaps, ordinary_record.gaps);
                assert_eq!(
                    artifact
                        .options
                        .modular
                        .sampling
                        .as_ref()
                        .unwrap()
                        .nb_repeats,
                    ordinary
                        .options
                        .modular
                        .sampling
                        .as_ref()
                        .unwrap()
                        .nb_repeats
                );
            }
            let strict = prepare(
                &params,
                &image,
                &base,
                NativeBackendOptions {
                    acceptance: NativeAcceptance::RequireExact,
                    ..settings
                },
                kind,
            );
            if changed {
                assert!(strict.is_err());
            } else {
                assert!(strict.is_ok());
            }
        }
        assert_eq!(params.learning_fraction, fraction);
    }
}
