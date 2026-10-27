# Native Multiframe Analysis

Start with the [practical analysis guide (Chinese)](MULTIFRAME_ANALYSIS_GUIDE.md)
for report interpretation, example calls and deciding when more search is not
worth its cost. This page is the detailed API/field reference.

`rs_jxl::multiframe::analyze` inspects whether a sequence of full-image states
has useful spatial reuse. It does not compress anything. Difference scanning,
reference selection, mask eligibility and rectangle proposals run in Rust.
The C ABI and Python API only marshal inputs and serialize the report.

This is separate from `encode_auto`: automatic encoding tests real candidate
payloads and retains the smallest. Analysis cannot prove that a multilayer JXL
is smaller than independently encoded images, nor predict encode/decode time.

## Rust

```rust
use rs_jxl::{Image, Result, codec::Encoder};
use rs_jxl::multiframe::{AnalysisOptions, AnalysisReport, Reference};

fn inspect(images: &[Image]) -> Result<AnalysisReport> {
    let mut encoder = Encoder::new(4)?;
    encoder.analyze_multiframe(images, &AnalysisOptions {
        reference: Reference::SmallestCrop,
        history_window: 8,
        extra_tile_sizes: vec![16, 64],
        threads: 4,
        ..Default::default()
    })
}
```

For a one-shot call use `multiframe::analyze(images, &options)`. A reusable
encoder context retains its thread pool; the analysis thread count must match.
One thread runs on the caller without creating a pool. Analysis needs no
animation metadata, duration, effort, or encoder search settings.

## Python

```python
from rs_jxl import Encoder, analyze_multiframe

report = analyze_multiframe(
    ["state0.png", "state1.png", "state2.png"],
    reference="auto",
    threads=4,
    analysis_options={
        "tile_size": 32, "max_rectangles": 32,
        "extra_tile_sizes": [16, 64], "history_window": 8,
    },
)

with Encoder(threads=4) as encoder:
    report = encoder.analyze_multiframe(["state0.png", "state1.png"])

for frame in report["frames"][1:]:
    stats = frame["previous_stats"]
    print(frame["index"], stats["changed_ratio"], stats["bbox_ratio"])
    for candidate in frame["candidates"]:
        print(candidate["strategy"], candidate["layer_count"],
              candidate["layer_area_ratio"])
    best = frame["candidates"][frame["geometry"]["best_area_candidate_index"]]
    print("minimum submitted area", best["layer_pixels"], best["layer_count"])
    for opportunity in frame["history"]:
        print("historical input", opportunity["reference_index"],
              opportunity["stats"]["changed_ratio"],
              opportunity["improves_static_bbox"])
```

Paths, encoded image bytes and NumPy/Torch inputs use the binding's normal
image transport. Inputs are never silently padded, resized or converted to a
common channel layout. They must have matching canvas dimensions, sample layout
and metadata. Explicit input metadata controls and extra-channel inputs are
available; see [Python API](PYTHON_API.md).

## Controls

| Native field | Default | Meaning |
| --- | --- | --- |
| `reference` | `SmallestCrop` | Candidate references: full, previous, first, or both. Python uses `"auto"` for `SmallestCrop`. |
| `allow_mask` | `true` | Include binary-alpha mask proposals when exactly eligible. |
| `allow_multi_rect` | `true` | Include multiple nonoverlapping REPLACE rectangles. |
| `adaptive_rectangles` | `true` | Include the hierarchy/merge proposal in addition to legacy tiled regions. |
| `tile_size` | `32` | Positive initial grid size in pixels. |
| `extra_tile_sizes` | `[]` | Additional positive grid sizes for multi-rectangle proposals. Empty keeps the original candidate set; duplicate grids/equivalent plans are not repeated. |
| `max_rectangles` | `32` | Positive layer cap for a multi-rectangle proposal. |
| `history_window` | `0` | Number of preceding input positions to inspect for historical opportunities, skipping first/previous comparisons already performed. Zero disables this extra work. |
| `rectangle_bytes_per_pixel` | `1.0` | Finite, nonnegative cost estimate used only by adaptive rectangle merging. |
| `threads` | `1` | Worker count, 1 through 256. |

Python supplies `reference` and `threads` as explicit keywords; other controls
belong in `analysis_options`. Unknown or misplaced controls are errors.

The real auto encoder measures its merge cost from the first full frame's
encoded payload. Analysis has no such measurement. Its adaptive rectangles can
therefore differ unless the caller provides that same cost estimate. Mask and
region eligibility and proposal code are shared with the encoder, not copied
implementations. Analysis does not accept trial limits or a pruning switch
because it executes no encoding trials.

Additional grids affect candidate geometry only when `allow_multi_rect=true`.
They use the same optional-scale planner as automatic encoding. Every grid must
still be positive when multiple-rectangle proposals are disabled. An extra grid
does not change a strategy name: a `tiles_previous` candidate can have
`tile_size=16` while another has `tile_size=32`. Full/crop/mask candidates have
`tile_size=null`. Increasing the number of grids can increase proposal work and
the number of real trials if the same options are later used for encoding.

## Report Semantics

Every ratio is a fraction in `[0, 1]`, not a percentage or compression ratio.
Multiply by 100 only for presentation.

- `frames[i].previous` and `.first` preserve the original bbox-only API. Identical
  comparisons are `null`; frame zero reports the full canvas in both fields.
- `previous_stats` and `first_stats` contain `changed_pixels`, `changed_ratio`,
  `bounds`, `bbox_pixels`, `bbox_ratio`, `bbox_fill_ratio` and `mask_eligible`.
  Frame zero has no comparison, so both statistics are `null`.
- `changed_pixels` counts the union of changed canvas positions, not the sum of
  changed channels. Extra-channel cells are mapped onto their covered canvas
  area. All sample words participate, including RGB below alpha zero, signed
  float zero and NaN payload bits. Identical float NaNs are not falsely changed.
- `bbox_fill_ratio = changed_pixels / bbox_pixels`; identical frames use zero.
  A large bbox with low occupancy is a reason to examine mask/multi-rectangle
  candidates, not proof that either will compress better.
- `plan` is the single-bbox REPLACE plan for the selected reference policy. It
  matches `multiframe::plan`, including ties preferring previous and the 1x1
  repeated-sample placeholder for an identical state. `reference_index` is the
  logical input index; `source` is the codestream reference slot (1 first,
  2 previous, 0 full replacement).
- `candidates` describes full, crop, eligible mask and multiple-rectangle
  proposals for the selected reference policy. It reports rectangles,
  `tile_size`, `layer_count`, submitted `layer_pixels` and `layer_area_ratio`. Masks still
  submit the entire bbox; changed-pixel count is not their submitted area.
  The layer count is the number of regular image layers, not any private patch
  reference frames an encoder might later insert.
- `frames[i].history` contains optional exact comparisons to older logical inputs.
  These observations do not add candidates or change `plan` or `reference_modes`.
- `frames[i].geometry` reports the minimum-area candidate and total proposal
  workload. Top-level `geometry` aggregates those minima, including multi-rect
  candidates, independently of the existing single-bbox reference summaries.
- `reference_modes` compares full/previous/first/smallest-crop single-bbox
  policies. Total area includes the first full frame and identical-state
  placeholders. `average_changed_ratio` excludes the initial frame and uses
  the selected reference; full replacements count as 100% new pixels.
- `best_geometric_reference` minimizes only that total bbox area. It does not
  select mask or multi-rectangle candidates, predict byte sizes, or establish
  that a frame sequence is preferable to independent images.
- `suitability` is a coarse geometric label: subsequent-frame mean submitted
  area below 30% is `strong`, below 70% is `moderate`, otherwise `limited`.
  One input is `single_frame`. These thresholds are advisory, not encoder
  switches, and never discard an encoder candidate.
- `crop_supported=false` and `full_frames_required` identify subsampled extra
  channels. Exact statistics are still reported, but encoding keeps full frames
  because decoder upsampling precedes compositing. Aligned crops alone do not
  preserve that interpolation boundary.
- `analysis_seconds` excludes input decoding, context construction and report
  serialization. Rust exposes the same time as `analysis_elapsed: Duration`.

Mask eligibility currently requires both images to be fully opaque integer
GA/RGBA at up to 16 bits, with unassociated primary alpha and no additional
extra planes. Analysis never creates synthetic alpha for RGB, ignores hidden
RGB, or suggests BLEND for partial transparency.

## Historical Opportunities

For logical input `i`, `history_window=w` considers the preceding positions
`[i.saturating_sub(w), i)`. It skips `0` and `i-1` because first/previous were
already analyzed, and returns the remaining entries newest first. Examples:

| Current index | Window | Additional `history` reference indices |
| ---: | ---: | --- |
| 5 | 0 or 1 | `[]` |
| 5 | 2 | `[3]` |
| 5 | 4 | `[3, 2, 1]` |
| 5 | 100 | `[3, 2, 1]` |
| 2 | 100 | `[]` |

The window is a positional distance, not a requested count of extra comparisons.
Top-level `history_window` echoes the requested value, even when the sequence
is too short to fill it. No unrequested hidden history cap is applied.

Each entry has:

- `reference_index`: the historical **logical input index**, never a slot ID.
- `stats`: the complete `DiffStatistics` object with the same exact sample-word
  comparisons as first/previous, including hidden RGB, extra channels, float
  signed zeros and NaN payload bits.
- `improves_static_bbox`: true only if cropping is supported and this reference
  has a strictly smaller `bbox_pixels` than **both** first and previous. This
  always compares against those two baselines, independent of `reference`.
  Equal bbox area is not an improvement, even if changed-pixel count is lower.

Identical states have `bbox_pixels=0`; actual encoding still requires its legal
1x1 repeat placeholder. If `crop_supported=false`, history still reports exact
observations, but `improves_static_bbox` is always false and all candidate plans
remain full-frame replacements.

**Historical opportunities are not actually available decoder reference slots.**
The encoder has a bounded set of saved canvases; an old matching input may have
already been evicted. The analyzer neither assigns a slot to history entries
nor simulates byte-selected retained-reference/lookahead decisions. A promising
entry identifies a reason to investigate reference retention, not a valid new
source slot or proof of compression savings. Increasing this window alone does
not change automatic encoding behavior.

## Candidate Workload

`frames[i].geometry` uses these fields:

| Field | Meaning |
| --- | --- |
| `best_area_candidate_index` | Index into that frame's `candidates`, minimizing `(layer_pixels, layer_count, canonical index)`. |
| `candidate_count` | Number of proposed candidates, after the shared planner's deduplication. |
| `trial_pixels` | Sum of submitted rectangle areas across **all** candidates, including masks and repeated rectangles. |
| `trial_layers` | Sum of regular-layer counts across all candidates. |
| `unmasked_rectangles` | Exact-rectangle repetition diagnostics, defined below. |

These `trial_*` fields describe hypothetical work before encoding/pruning. No
trial has actually been run, and no encoding time is measured by this API.
Geometry-only ordering is not compression ordering. The encoder can prefer a
larger rectangle that costs fewer bytes, or prune candidates before finishing.
Regular-layer counts exclude any private patch reference frames that an encoder
might insert. Additional layers also impose decode/header costs, not quantified
by their pixel area.

`unmasked_rectangles` identifies equal `(x, y, width, height)` rectangles within
one input frame, independent of their reference slot or tile-grid provenance:

| Field | Meaning |
| --- | --- |
| `total_pixels`, `total_layers` | Area and occurrences of every unmasked rectangle across candidates. |
| `unique_pixels`, `unique_rectangles` | Area and count of distinct rectangle keys. Area is a sum, **not** the spatial union of overlapping rectangles. |
| `repeated_pixels`, `repeated_layers` | Area and occurrences after the first occurrence of each key. |

`total_pixels = unique_pixels + repeated_pixels`, and
`total_layers = unique_rectangles + repeated_layers`. Mask candidates contribute
to `trial_*` but are entirely excluded from this reuse accounting: their pixel
payload depends on the compared reference, even for identical bounds. Partially
overlapping rectangles are not identical keys and are not counted as reusable.

This is an **upper bound on potentially reusable unmasked work**, not a cache-hit
prediction. Actual patch plans can bypass the cache; pruning can prevent trials
from reaching a rectangle or completing a reusable payload. Reference headers
must still be regenerated. The analyzer has no encoder settings and cannot infer
those decisions or compressed payload bytes.

## Aggregate Geometry

Top-level `geometry` contains:

- `layer_pixels`, `layer_area_ratio`, `layer_count`: sums of the per-frame
  minimum-area candidates, including the initial full canvas and any 1x1 repeat
  placeholders. The ratio divides by `width * height * frame_count`.
- `crop_frames`: inputs for which that minimum candidate submits less than one
  full canvas. `multi_rect_frames` counts minimum candidates with multiple layers.
- `suitability`: the same area thresholds as `reference_modes`, but using these
  multi-candidate minima. It excludes the initial full canvas when applying the
  30%/70% thresholds. Single input and subsampled-extra restrictions retain the
  `single_frame` / `full_frames_required` semantics.
- `candidate_count`, `trial_pixels`, `trial_layers`, `unmasked_rectangles`: sums
  of the corresponding per-input diagnostics. Reuse is never inferred between
  different logical input frames, even if their rectangle coordinates match.

For two changed pixels far apart, the single bbox may cover most of the canvas
and be labeled `limited`, while a two-rectangle proposal submits only two pixels
and is labeled `strong`. Both reports are valid and intentionally remain separate.
The new summary does not rewrite `reference_modes`, `best_geometric_reference`,
or the compatibility single-bbox plan. Nor does it imply the multiple-layer
candidate is smaller or faster to encode than the bbox candidate.

All area/count accumulations use checked arithmetic; overflow is an analysis
error rather than a wrapped workload estimate. No compressed byte estimate is
added to any result. `rectangle_bytes_per_pixel` remains only the caller's
existing adaptive-merge cost parameter, not an observed or predicted size.

## Comparing Actual Compression

To establish a byte saving, measure at least these three routes at the same
effort and input pixel format:

1. Sum of independently encoded JXL images.
2. Full-size multiframe replacement without explicit differencing.
3. Crop and/or automatic multiframe encoding.

Keep JPEG byte-reversible transcode separate from JPEG decoded to pixels and
then encoded losslessly. They preserve different things and have different
costs. See [native multiframe measurements](MULTIFRAME_NATIVE_BENCHMARK.md) for
examples where extra rectangles reduce bytes but increase layer/decode costs.

## Existing Fixture Check

One release-DLL call per set on Windows, 2026-10-27, reusable four-worker context,
default analysis controls. These are observations, not stable benchmark averages.
No images were encoded or written. Times exclude Python import and context
creation; wall time includes reading/decoding the source files and FFI transport.

| Set | Input states | Native analysis | End-to-end call |
| --- | --- | ---: | ---: |
| t1 | 4 PNGs, 4096x4096 RGBA | 51.84 ms | 899.17 ms |
| t2 | 4 JPEGs, 2000x2183 RGB | 14.21 ms | 251.51 ms |

For t1, first-reference crop area including the initial full frame is 36.94% of
the all-full sequence area, versus 38.75% with previous references. The final
state differs at 5.37% of canvas positions from its predecessor, but its bbox
covers 53.71%. A first-reference merged proposal submits 5.35% of the canvas
through 32 rectangles at the default cost estimate. These values expose sparse,
dispersed changes; they are not compressed-size ratios.

For t2, previous-reference bboxes each cover 75.20% of the canvas. The smallest
area proposals for the three later states cover 45.70%, 26.72% and 45.29%, with
32, 31 and 32 layers. RGB inputs are correctly ineligible for the existing-alpha
mask route. Per the measured encoding results linked above, extra layers can
cost substantial encoding and decoding time despite smaller geometric area.

The complete t3 sequence is rejected for incompatible image geometry/metadata;
the analyzer does not silently pad the differently sized frame.

## Implementation and Tests

The implementation is in `crates/rs-jxl/src/multiframe/analysis.rs` and its
`analysis/history.rs` and `analysis/geometry.rs` submodules, with shared `diff.rs`,
candidate geometry in `auto.rs`/`scales.rs`, and the existing rectangle planners.
At most two static reference maps are retained per input; the second frame shares
its first/previous map. History scanning creates and releases one additional map
at a time. It does not retain all historical bitmaps, cropped images or encoded
payloads. Reports retain geometry and statistics only.

Exact scanning is linear in compared samples per comparison. Increasing the
history window increases the number of scans and stored statistics, up to
quadratic comparison count for an unbounded sequence-wide window. It is opt-in,
not a free extension of decoder reference capacity. Rectangle proposals add
work proportional to occupied tiles and requested distinct scales. Workload
diagnostics use ordered rectangle keys and never rescan image samples.

`crates/rs-jxl/tests/multiframe_analysis.rs` covers exact statistics, all reference
policies, hidden RGB, float words, wide integer samples, extra-channel unions,
subsampling restrictions, identical states, worker reuse and invalid controls.
It also compares reported candidate areas and layer counts with actual encoder
trials using the same measured merge cost and verifies the decoded final state.
Additional coverage includes history-window boundaries, strict bbox improvement,
subsampled history restrictions, hidden/float/extra history samples, multiscale
deduplication, sparse-region versus bbox suitability, canonical area/layer ties,
mask exclusion, per-input reuse scope and checked-counter overflow. FFI tests in
`crates/rs-jxl-ffi/src/analysis.rs` check the full JSON projection and that analysis
returns no encoded data. Central test execution is handled by the parent task;
these additions do not themselves assert a new benchmark or conformance result.
