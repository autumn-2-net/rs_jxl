# Frame-Aware Lossless Search

`scheduler::run_native_sequence_search` applies the frozen source pixel search
to each logical frame while retaining its real frame context. It does not join
independently encoded JXL files or re-encode the selected winners.

```rust
use rs_jxl::{EncodeOptions, Image, frame::Frame, scheduler::{
    CandidateParams, DriverOptions, NativeBackendOptions,
    run_native_sequence_search,
}};

fn search(canvas: &Image, frames: &[Frame<'_>]) -> rs_jxl::Result<Vec<u8>> {
    let result = run_native_sequence_search(
        canvas, frames, &CandidateParams::default(), &DriverOptions::default(),
        &EncodeOptions { threads: 4, ..Default::default() },
        NativeBackendOptions::default(),
    )?;
    Ok(result.bytes)
}
```

The canvas supplies dimensions, color/extra-channel metadata and animation
timing. Each frame retains its origin, duration (including zero), timecode,
blend mode, saved reference and name. Reference availability is validated for
the entire sequence before search. Source patches reserve reference slot 3.
As with the source pixel driver, explicit nondefault overrides can make input
ineligible for search. For a sequence requiring slot 3, use `encode_frames`
with patches disabled rather than silently changing its reference semantics.

Candidate size comparisons include the actual frame header and any private
patch-reference frame. Crop probes remain separate estimates. Final winners
are appended after one shared image header and wrapped in one container. Every
frame report includes actual backend options, candidates, adaptations and its
encoded byte count. The byte counts exclude the shared image/container header.

One owned worker pool is shared across all frames and candidates. Full-frame
count and pixel-work ceilings apply to the sequence as a whole. The output
limit includes the final shared headers. Probe limits apply to each frame.
Limits are errors, never silent effort reduction or partial search success.

This API does not infer deltas. `multiframe::plan` can prepare exact full-frame,
previous-reference or first-reference crop sequences. It compares every
channel, including hidden RGB, and uses REPLACE rather than alpha compositing.
Explicit alpha-blended layers remain supported with the caller's semantics.

Acceptance compares decoded pixels through both official decoders. The current
integration suite `searched_layers_and_animation_match_official_djxl` covers
8/16-bit RGBA, hidden samples, animation, zero-duration layers and explicit
reference-only frames. See `PORT_VERIFICATION.md` for executed test results.
