# Byte-Selected Automatic Frames

`multiframe::encode_auto` encodes a sequence of complete image states without
requiring callers to calculate residuals. It compares actual encoded layer bytes,
not bbox area or estimated entropy, and retains the winner without re-encoding.
The older `multiframe::plan` remains a cheap geometry-only API with unchanged
semantics; `Reference::SmallestCrop` there still means smallest area.

```rust
use rs_jxl::{EncodeOptions, Image};
use rs_jxl::multiframe::{encode_auto, AutoOptions, Reference};

fn archive(images: &[Image], durations: &[u32]) -> rs_jxl::Result<Vec<u8>> {
    let result = encode_auto(images, durations,
        &EncodeOptions { threads: 4, ..Default::default() },
        &AutoOptions { reference: Reference::SmallestCrop, ..Default::default() })?;
    Ok(result.bytes)
}
```

Inputs must have identical geometry, channel layout and image metadata. Animation
timing is declared in `ImageMetadata::animation`; durations are ticks in that
timebase. With no animation metadata, durations must all be zero. The function
does not add an alpha channel or convert color spaces/sample formats. It returns
bytes in memory. Files/writers can consume those bytes using their own I/O policy.

## Candidates

Every logical frame tests a full-frame REPLACE candidate. Subsequent frames can
also test the following, against the previous image and/or the first image:

- Crop REPLACE: the exact bounding rectangle of all changed samples.
- Binary alpha mask: changed pixels retain their current values; unchanged
  pixels become zero with alpha zero. BLEND references the chosen full image.
- Multiple REPLACE rectangles: occupied tiles are tightened to changed samples
  and adjacent equal spans are merged. Each rectangle is an ordinary JXL layer;
  intermediate layers have duration zero, and only the final layer carries the
  logical frame's duration.

All color and extra samples participate, including invisible RGB and floating
sample words. Identical frames use a 1x1 exact replacement, not an illegal empty
layer. A permanent first-image reference uses slot 1; the previous reconstructed
state and intermediate rectangle states use slot 2. Slot 3 stays available for
the existing intra-frame patch encoder. References store reconstructed full
states, not residual rectangles. Decoders perform reconstruction normally.

Mask eligibility is deliberately strict: existing opaque integer GA/RGBA,
8/16-bit storage, only its primary alpha extra channel, and no premultiplied
association. Partial transparency, hidden RGB, floating data, multiple alpha or
other extras use REPLACE candidates. Alpha precision and global metadata are
unchanged; this is a binary-valued plane, not a global alpha-depth rewrite.
Subsampled extras use full frames because upsampling before compositing can
change crop-boundary samples. No unsupported blend semantics are approximated.

## Selection and Reports

`Reference::Previous` tests previous-image candidates, `First` tests first-image
candidates, `SmallestCrop` tests both, and `Full` disables differences. Full is
always considered. Each frame report contains all eligible candidates, encoded
bytes, regular-layer count, elapsed encode time and `winner_index`. Candidate
bytes include frame headers and private patch-reference frames, but not shared
image/container headers. `encoded_frames` counts regular layers only.

Selection is the smallest measured byte count; equal sizes keep the earlier
full/crop candidate. Time is measured, not used as a hidden lossless-effort
downgrade. This is a per-frame minimum over a bounded candidate set, not a proof
of the globally smallest JXL. It does not run the full source-parameter search
again inside each candidate. Pass explicit `EncodeOptions` for the desired
backend tools. One owned worker pool is shared across candidate encodes.

By default, tiles are 32 pixels and at most 32 rectangles are proposed. A tile
candidate exceeding that bound is not admitted; full/crop remain available.
Input frame, candidate-count and cumulative candidate-pixel budgets fail
explicitly when exceeded. Output limits include the final container. These
budgets do not claim to bound total process RSS or wall-clock time.

## Reading Frames

Normal decoding returns coalesced visible animation frames. `decode_frame` uses
their zero-based indices and reconstructs reference dependencies. True
zero-duration input layers remain zero-duration: they do not become individually
visible animation frames. Raw decoding (`coalescing = false`) exposes the
generated rectangles/masks, not one original full image per layer. The winning
report's regular-layer counts give the mapping of logical inputs to raw layers;
the normal codestream does not embed this application-level report. Keep it
separately when that mapping is needed. This API is for complete image states;
use `encode_frames` for caller-defined layer names, blend instructions or timecodes.

## Verification

`tests/multiframe_auto.rs` checks typed samples, hidden RGB, mixed/zero timing,
first/previous references, masks, multiple rectangles, extra channels and resource
errors. `metadata_frame_reference::automatic_byte_selected_layers_match_official_decoders`
is an opt-in independent djxl comparison using `RS_JXL_DJXL`. Correctness means
decoded samples, never equality of compressed JXL bytes. Execution results are
recorded separately after the integrated test run.

### 2026-10-24 Checkpoint

The ten automatic-strategy integration tests passed in release, including
floating special words, opaque/partial/premultiplied alpha, extra planes,
mixed durations and one/four-thread execution. Twelve CLI tests also passed
in the integrated run. The bounded tile-coverage unit test checks nonoverlap
and coverage at tile sizes 1/8/32/128.

Official djxl 0.12 and 0.13 passed **96 independent exact-sample comparisons**.
The matrix exercised 20 actual mask winners and 40 multiple-rectangle winners,
not merely candidates that were never decoded. It covers 8/16-bit RGBA,
invisible RGB, first/previous/both references, visible animation and mixed
zero-duration layers. These are functional results, not a throughput benchmark
or proof of optimal compressed size across every possible decomposition.

### Additional Edge Regressions (2026-10-24)

`tests/multiframe_auto.rs` additionally covers:

- Extra-only changes across six logical states with mixed zero/nonzero
  durations, requiring a multiple-rectangle winner. Coalesced and selected-frame
  extra samples are checked directly against the inputs. Raw layers check the
  first/previous slots, extra-channel blending sources, final-layer timing and
  the report-to-layer mapping for all three reference policies.
- Actual opaque-mask winners with color/alpha depths of 8/16, 16/8 and 12/1.
  Original color samples and the authoritative native alpha plane are checked
  separately; interleaved decoded alpha follows the color output depth.
- Inclusive logical-frame, trial-count and cumulative trial-pixel limits,
  per-image pixel limits, exact final output size including the Level 10
  container, and rejection of mismatched image depths.

The shared verification helper also checks the first minimum-byte candidate on
ties and exact canvas dimensions. All 13 tests in this integration target now
pass in release, including these additions. Output-budget checks compare size
and decoded samples, not identity of compressed bytes.

The expanded independent matrix subsequently passed **240 exact-sample
comparisons** against djxl 0.12/0.13, including color/alpha depths 8/8, 16/16,
8/16, 16/8 and 12/1. It exercised 50 actual mask winners and 100 actual
multiple-rectangle winners. Native ranges are checked from each channel's
declared depth, not inferred from the U8/U16 storage type. Extra-only state
coverage above uses the Rust decoder; it is not counted as an independent djxl
comparison in this expanded matrix.
