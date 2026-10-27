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
- Adaptive merged rectangles (`MergedPrevious`/`MergedFirst`): an additional
  cost-guided proposal when `allow_multi_rect` and `adaptive_rectangles` are
  enabled. The estimate is anchored by the initial full frame's measured encoded
  bytes per pixel. It proposes a partition only; its estimated cost is never
  reported as encoded size or used to select the winner. All legacy candidates
  remain eligible, with new proposals appended after them for stable ties.

Rust builds a shared exact difference map per analyzed input/reference pair.
Bounds, changed-pixel counts, mask construction and rectangle proposals reuse
that analysis. Python and CLI adapters neither scan pixels for differences nor
implement the candidate/pruning algorithms.

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
always considered. Each `AutoFrameReport` contains:

- `candidates`: only completed `AutoCandidateReport` records, in canonical order,
  with actual `encoded_bytes`, fully encoded regular-layer count `encoded_frames`,
  `trial_pixels` and native `elapsed`. Candidate bytes include frame headers and
  private patch-reference frames, but not shared image/container headers.
- `winner_index`: an index into the completed `candidates` list, not execution
  order or a combined completed/pruned list.
- `pruned`: separate `AutoPrunedReport` records with `strategy`, `lower_bound`,
  `encoded_frames`, `planned_frames`, `trial_pixels` and `elapsed`. A lower bound
  is never a completed size. `encoded_frames` counts fully finished regular
  layers only, excluding an interrupted layer and private reference-only frames.
- `analysis_elapsed`: native elapsed difference/planning time.
- `differences`: `AutoDiffReport` entries containing zero-based input `reference`,
  exact `changed_pixels`, and `bounds: Option<Rect>`. No change has zero pixels
  and `None` bounds; no analyzed references gives an empty list. A reference
  index is not a JXL reference-slot number.

`trial_pixels` counts areas actually submitted to trial encoding, including an
interrupted layer, not analysis scans. Time fields are native `Duration`s; the
FFI exposes seconds as `elapsed_seconds` and `analysis_seconds` without Python
timing or recomputation.

`prune=true` (the default) ranks inexpensive work first, then stops a losing
trial only when finalized output gives a provable byte lower bound that cannot
win under the canonical tie rule. An estimate cannot justify skipping a
candidate. Reports are restored to canonical order afterward. Disabling pruning
with `prune=false` completes all eligible trials or fails explicitly on an error.
For the same candidate set and sufficient budgets, pruning preserves selection;
it does not change eligibility or the preference among equal-byte candidates.

Selection is the smallest measured byte count; equal sizes keep the earlier
full/crop candidate. Time is measured, not used as a hidden lossless-effort
downgrade. This is a per-frame minimum over a bounded candidate set, not a proof
of the globally smallest JXL. It does not run the full source-parameter search
again inside each candidate. Pass explicit `EncodeOptions` for the desired
backend tools. One owned worker pool is shared across candidate encodes.

By default, tiles are 32 pixels and at most 32 rectangles are proposed. A tile
candidate exceeding that bound is not admitted; full/crop remain available.
The adaptive proposal instead coarsens occupied tiles to fit this explicit
rectangle budget. It builds a tile-aligned binary hierarchy, then uses a
sibling-merge heap. Its cost is area times the first frame's encoded bytes per
pixel plus 64 heuristic bytes per layer; 64 is not a normative header size.
The planner is not an exhaustive partition search. Allocation is checked, and
there is no quadratic all-pairs rectangle search.
Both geometry controls must be positive; `max_rectangles` has no additional
4096 cap. `allow_mask`, `allow_multi_rect`, `prune` and `adaptive_rectangles`
default to true. Set `adaptive_rectangles=false` to retain legacy rectangle
proposals without the new merged proposal; this is independent of pruning.
`allow_multi_rect=false` disables both tile and adaptive multi-rectangle proposals.

Smaller output can cost more encoding and decoding time: each selected rectangle
is a real layer, and coalescing many layers can require repeated full-canvas work.
Selection does not optimize decoder latency. Keep pruning but set
`adaptive_rectangles=false` for the old candidate set with less repeated work;
use an explicit smaller `max_rectangles` to explore a different size/layer tradeoff.
Neither setting is a hidden codec limit or an effort change.

Input frame (`max_frames`), candidate-count (`max_total_trials`) and cumulative
candidate-pixel (`max_trial_pixels`) budgets use native unlimited defaults
(`usize::MAX`, `usize::MAX`, `u64::MAX`). Explicit zero is accepted and refuses
work, not an alias for unlimited. Limits fail explicitly when exceeded; pruning
does not turn a work-budget error into a partial success. Existing output limits
include the final container. These budgets do not claim to bound total process
RSS or wall-clock time; no new codec limits are introduced here.

The correctness contract is unchanged: original decoded samples, including
invisible RGB, must match exactly. JPEG inputs to the pixel multiframe path are
pixel-lossless relative to the input decoder, not byte-reversible JPEG archives.
Use the separate JPEG transcode/reconstruction API for original-byte recovery.

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

### 2026-10-27 Native Update

Validated on Windows x86-64 with the release CLI and FFI DLL built from this
update. Algorithms live in Rust: `multiframe/diff.rs` owns exact shared change
coverage, `regions.rs` owns masks and legacy tiles, `partition.rs` owns adaptive
geometry, and `auto.rs` owns trial selection and reporting. CLI/Python forward
controls; neither implements these algorithms.

- 29 multiframe unit tests, 13 automatic-frame integration tests, 7 pruning
  integration tests and 8 reusable-context tests passed.
- 15 section-budget tests passed; their opt-in official-decoder case was not
  included in that run. The non-layout independent-group regression now checks
  bounded cancellation and complete-output sample recovery.
- All 79 non-ignored CLI tests and 26 FFI unit/integration tests passed; two
  unrelated opt-in CLI tests remained ignored.
- All 5 opt-in `metadata_frame_reference` tests passed against official djxl
  0.12 and 0.13. This includes 240 existing automatic-strategy comparisons and
  16 new comparisons requiring a merged-rectangle winner at 8/16 bits, with
  both visible frames and zero-duration layers.
- Python bindings: 173 passed with official-decoder and CUDA-transfer checks
  enabled; benchmark harness: 10 passed. Two preexisting packaging warnings
  were non-fatal.
- Workspace all-target compile checks, strict Clippy and formatting passed.

See [MULTIFRAME_NATIVE_BENCHMARK.md](MULTIFRAME_NATIVE_BENCHMARK.md) for the
small real-image e9 comparison and supplementary e5 check, including independent
still-image sums, bpp, encode/decode times and latency regressions. These
tests are not a full-corpus or cross-platform runtime acceptance claim.

`tests/multiframe_auto.rs` checks typed samples, hidden RGB, mixed/zero timing,
first/previous references, masks, multiple rectangles, extra channels and resource
errors. `metadata_frame_reference::automatic_byte_selected_layers_match_official_decoders`
is an opt-in independent djxl comparison using `RS_JXL_DJXL`. Correctness means
decoded samples, never equality of compressed JXL bytes. Execution results are
recorded above and in the benchmark report.

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
