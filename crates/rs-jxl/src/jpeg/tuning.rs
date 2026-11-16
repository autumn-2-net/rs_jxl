//! Explicit coefficient-encoder experiments; no changes to reconstruction or
//! decoder syntax. Ordinary encoding and source-search retain their defaults.

use super::{TranscodeOptions, archival, cfl::CflFitPolicy, transcode as coefficients};
use crate::Result;

#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub cfl_fit: CflFitPolicy,
    /// Native symmetric MA and half description cost on learned auxiliary trees.
    pub symmetric_ma: bool,
    /// Merge adjacent luma DC buckets only for AC statistical conditioning.
    pub paired_luma: bool,
}

/// Single-pass experiment, not a baseline/candidate tournament. Neither a better
/// local score nor fewer contexts guarantees a smaller complete JXL container.
pub fn transcode(input: &[u8], options: &TranscodeOptions, tuning: Options) -> Result<Vec<u8>> {
    let mut settings = options.resolve()?;
    settings.cfl_fit = tuning.cfl_fit;
    settings.symmetric_ma = tuning.symmetric_ma;
    settings.paired_luma = tuning.paired_luma;
    let jpeg = archival::read_input(input, &options.limits.input)?;
    coefficients::encode_coefficients(&jpeg, &settings)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let original =
                super::super::transcode_reference_tests::fixtures::fixture(name).unwrap();
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
                },
            ] {
                for threads in [1, 4] {
                    let options = TranscodeOptions {
                        effort: 9,
                        threads,
                        cooperative_ma: Some(Default::default()),
                        ..Default::default()
                    };
                    let encoded = transcode(&original, &options, tuning).unwrap();
                    let restored =
                        crate::decode::reconstruct_jpeg(&encoded, &Default::default()).unwrap();
                    assert_eq!(restored, original, "{name} {tuning:?} t{threads}");
                }
            }
        }
    }
}
