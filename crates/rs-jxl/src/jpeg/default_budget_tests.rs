use super::{
    TranscodeLimits, TranscodeOptions, coefficient_reader::CoefficientReadLimits,
    compatibility::ProbeLimits, dc_streams::PrepareLimits, reader::ReadLimits,
    reconstruction_decode::ReconstructionDecodeLimits, scan_writer::WriteLimits,
    transcode::Options,
};

fn assert_structural(limits: &ReadLimits) {
    assert_eq!(limits.max_input_bytes, usize::MAX);
    assert_eq!(limits.max_records, usize::MAX);
    assert_eq!(limits.max_scans, usize::MAX);
    assert_eq!(limits.max_quant_tables, usize::MAX);
    assert_eq!(limits.max_huffman_tables, usize::MAX);
    assert_eq!(limits.max_restart_markers, usize::MAX);
}

fn assert_coefficient(limits: &CoefficientReadLimits) {
    assert_structural(&limits.structural);
    assert_eq!(limits.max_coefficient_bytes, usize::MAX);
    assert_eq!(limits.max_block_visits, usize::MAX);
    assert_eq!(limits.max_hint_entries, usize::MAX);
    assert_eq!(limits.max_padding_bits, usize::MAX);
}

fn assert_write(limits: &WriteLimits) {
    assert_structural(&limits.structural);
    assert_eq!(limits.max_output_bytes, usize::MAX);
    assert_eq!(limits.max_block_visits, usize::MAX);
    assert_eq!(limits.max_hint_entries, usize::MAX);
    assert_eq!(limits.max_padding_bits, usize::MAX);
    assert_eq!(limits.max_refinement_bits, usize::MAX);
}

fn assert_transcode(options: &Options) {
    assert_eq!(options.max_output_bytes, usize::MAX);
    assert_eq!(options.max_coefficients, usize::MAX);
    assert_eq!(options.max_tokens, usize::MAX);
    assert_eq!(options.sample_limit, usize::MAX);
}

#[test]
fn defaults_add_no_refusal_budgets_at_any_jpeg_entry_point() {
    assert_structural(&ReadLimits::default());
    assert_coefficient(&CoefficientReadLimits::default());
    assert_write(&WriteLimits::default());
    let probe = ProbeLimits::default();
    assert_eq!(probe.max_input_bytes, usize::MAX);
    assert_eq!(probe.max_markers, usize::MAX);
    assert_eq!(probe.max_scans, usize::MAX);
    assert_eq!(probe.max_findings, usize::MAX);
    let prepare = PrepareLimits::default();
    assert_eq!(prepare.max_samples, usize::MAX);
    assert_eq!(prepare.max_output_bytes, usize::MAX);
    let decode = ReconstructionDecodeLimits::default();
    assert_eq!(decode.max_jbrd_bytes, usize::MAX);
    assert_eq!(decode.max_metadata_bytes, usize::MAX);
    assert_eq!(decode.max_coefficient_values, usize::MAX);
    assert_write(&decode.write);
    let transcode = TranscodeLimits::default();
    assert_coefficient(&transcode.input);
    assert_eq!(transcode.max_output_bytes, usize::MAX);
    assert_eq!(transcode.max_coefficients, usize::MAX);
    assert_eq!(transcode.max_tokens, usize::MAX);
    assert_eq!(transcode.sample_limit, usize::MAX);
    assert_transcode(&Options::default());
    assert_transcode(&TranscodeOptions::default().resolve().unwrap());
}

#[test]
fn explicit_transcode_budgets_are_preserved_and_zero_still_rejects() {
    let mut options = TranscodeOptions {
        limits: TranscodeLimits {
            max_output_bytes: 101,
            max_coefficients: 202,
            max_tokens: 303,
            sample_limit: 404,
            ..Default::default()
        },
        ..Default::default()
    };
    let resolved = options.resolve().unwrap();
    assert_eq!(resolved.max_output_bytes, 101);
    assert_eq!(resolved.max_coefficients, 202);
    assert_eq!(resolved.max_tokens, 303);
    assert_eq!(resolved.sample_limit, 404);
    options.limits.sample_limit = 0;
    assert!(matches!(options.resolve(), Err(crate::Error::Limit(_))));
}

#[test]
fn jpeg_modular_policies_have_no_default_selected_sample_ceiling() {
    for tier in -1..=9 {
        for decode_speed in 0..=4 {
            for options in [
                super::modular_policy::options_for_dc(tier, decode_speed),
                super::modular_policy::options_for_ac_metadata(tier, decode_speed),
                super::modular_policy::options_for_quant(tier, decode_speed),
                super::modular_policy::options_for_global_entropy(tier, decode_speed),
            ] {
                assert_eq!(options.unwrap().sample_limit, None);
            }
        }
    }
}
