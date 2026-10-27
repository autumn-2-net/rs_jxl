# Native Multiframe Analysis

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
    analysis_options={"tile_size": 32, "max_rectangles": 32},
)

with Encoder(threads=4) as encoder:
    report = encoder.analyze_multiframe(["state0.png", "state1.png"])

for frame in report["frames"][1:]:
    stats = frame["previous_stats"]
    print(frame["index"], stats["changed_ratio"], stats["bbox_ratio"])
    for candidate in frame["candidates"]:
        print(candidate["strategy"], candidate["layer_count"],
              candidate["layer_area_ratio"])
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
| `max_rectangles` | `32` | Positive layer cap for a multi-rectangle proposal. |
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
  `layer_count`, submitted `layer_pixels` and `layer_area_ratio`. Masks still
  submit the entire bbox; changed-pixel count is not their submitted area.
  The layer count is the number of regular image layers, not any private patch
  reference frames an encoder might later insert.
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

The implementation is in `crates/rs-jxl/src/multiframe/analysis.rs`, with shared
`diff.rs`, candidate geometry in `auto.rs`, and the existing rectangle planners.
Only two reference maps are live per input state, and the second frame shares
its first/previous map. Reports retain geometry and statistics, not image
copies or bitmaps. Exact scanning is linear in compared samples; adaptive
region construction adds work proportional to occupied tiles.

`crates/rs-jxl/tests/multiframe_analysis.rs` covers exact statistics, all reference
policies, hidden RGB, float words, wide integer samples, extra-channel unions,
subsampling restrictions, identical states, worker reuse and invalid controls.
It also compares reported candidate areas and layer counts with actual encoder
trials using the same measured merge cost and verifies the decoded final state.
