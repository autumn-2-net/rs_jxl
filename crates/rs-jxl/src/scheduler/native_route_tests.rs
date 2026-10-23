//! Projection boundary tests; no codec or optional parallel-tool implementation.
use super::*;
use crate::{PixelData, PixelLayout, scheduler::params::SourceBuffering};

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
