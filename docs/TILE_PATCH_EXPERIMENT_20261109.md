# Exact Tile Patch Experiment

## Scope and Provenance

Opt-in native Rust detector inspired by the fixed-grid repetition idea in
[Cygnus JxlEncoder.java](https://github.com/ebremer/cygnus/blob/6e630c15c59f535d2346e14807ef7cc5ab1fbf6d/JPEGXL/src/main/java/com/ebremer/cygnus/jpegxl/encoder/JxlEncoder.java),
`encodeWithPatches` / `tryPatches`. Inspected master resolves to
`6e630c15c59f535d2346e14807ef7cc5ab1fbf6d`; Cygnus identifies its license as
Apache-2.0. This implementation is independently written against the existing
Rust tools: no Java code, Java dictionary writer, external codec, C++ link or
runtime process is imported. Existing libjxl-derived tools retain their BSD
notices. Rust base before this experiment: `9050e6c`.

Cygnus groups exact 16x16 color tiles, requires at least three occurrences,
stores unique tiles in a reference frame and uses standard Replace patches.
Its smaller-output guarantee comes from encoding the plain alternative too.
Its full-canvas atlas and per-channel mean fill are not copied here: this
experiment packs a compact atlas and uses the existing zero-filled residual.
The reported Cygnus synthetic gain is not a comparison against our encoder or
official cjxl e9 and is not used as performance evidence here.

No decoder syntax, JPEG transcode, lossy policy, default effort routing or Python
ABI changes. The PR5007 MA and other experimental knobs remain disabled.

## Native Interface

```rust
use rs_jxl::{EncodeOptions, encode};
use rs_jxl::tools::patches::{TilePatchOptions, TilePatchReceipt};
use std::sync::Arc;

// image: an existing rs_jxl::Image; source options may be resolved beforehand.
let receipt = Arc::new(TilePatchReceipt::default());
let options = EncodeOptions {
    threads: 8,
    tile_patches: Some(TilePatchOptions {
        receipt: Some(receipt.clone()),
        ..Default::default()
    }),
    ..Default::default()
};
let bytes = encode(&image, &options)?;
let work = receipt.snapshot();
```

`tile_patches`, `patches` (component Replace), and `source_patches` (source Add)
are mutually exclusive. The direct API first performs cheap admission, then
encodes one route. It does not perform two full encodes to promise a size win.
`force=true` bypasses admission for diagnostics. The source
scheduler adapter clears the tile experiment while projecting its own policy;
set it afterwards for controlled direct-candidate experiments.

| TilePatchOptions field | Default | Meaning |
| --- | ---: | --- |
| off_grid | false | Opt-in arbitrary destination origins for grid-seeded exact tiles; see the [follow-up experiment](QLIC_LOCAL_DECISIONS_20261112.md) |
| max_discovery_positions | 262144 | Off-grid discovery guard; exhaustion declines the candidate, not the input |
| force | false | Bypass admission for controlled diagnostic comparisons only |
| min_estimated_savings | 0.02 | Finite 0..1; minimum net gain relative to a sampled coding-cost proxy |
| full_frame_min_estimated_savings | 0.10 | Finite 0..1; stronger minimum when a dictionary forces frame-area encoding into full-frame processing |
| tile_size | 16 | Positive square side, arithmetic/i32 checked; full non-overlapping grid anchored at (0,0) |
| min_occurrences | 3 | At least 2; exact repeated occurrences required |
| max_unique_tiles | 4096 | Positive retained dictionary limit, not a prefix sampling limit |
| max_positions | 65536 | Positive retained occurrence limit |
| max_reference_pixels | 4194304 | Positive atlas area budget, including padding |
| receipt | None | Optional invocation-owned work collector; reuse accumulates scans |

These are explicit heuristic budgets for this opt-in detector, not new codec
limits. They do not constrain ordinary pixel encoding. A declined/budget-pruned
tile remains unchanged in the ordinary residual. Input indexing still scans the
complete grid and uses O(number of tiles) storage; retained budgets are not a
total process memory cap. The retained position count also respects the existing
decoders' patch/blending capacity, including all metadata extra channels and
geometric allocation headroom. This matters for very small configurable tiles
with many extra channels; excess occurrences remain ordinary pixels.

Receipt counters report scanned tiles, constant tiles, repeated groups, retained
groups/positions, covered pixels and atlas pixels. The receipt's seconds cover
hash/group/atlas/residual preparation, excluding the outer Image-to-plane copy
and initial view validation. Overall native encode timing includes those costs.
Unsupported main depth returns without discovery; a zero call count is possible.
`admission_rejected` counts cheap screening rejections; estimated saved/baseline
bits are diagnostic proxy values, not measured output bits. A rejected plan has
zero retained groups/positions/atlas pixels even if repetitions were found.

### Cheap Admission

Before full residual allocation, reference-frame encoding, MA learning or the
entropy backend, `patch_tile_admission.rs` scores unique repeated tiles using
the cheapest of four fixed standard predictors (Zero/Left/Top/Gradient).
The score combines HybridUint token histogram entropy and extra bits. Only one
copy of each unique tile is retained in the reference. The screen subtracts a
conservative dictionary/header proxy and right/bottom prediction-boundary charge.
Nonpositive estimated gain is rejected immediately.

For a positive estimate, at most 128 stratified complete tiles estimate the
ordinary image cost. It is floored by the cost of the selected source regions;
partial edges are conservatively charged raw sample bits. The native frame
adapter raises the minimum gain from 2% to at least 10% if a real dictionary
would abandon eligible frame-area encoding. There is no file-name or dataset
exception and no change to the final codec tools for admitted candidates.

This is conservative heuristic admission, not a mathematical bound on the
full MA/LZ77 result. It can miss real wins or admit a size regression. It does
not claim that positive raw coverage means a win, or that sampled estimates
replace final-size validation. The goal is to reject weak routes before paying
for them, not to trial-encode both choices. The ordinary encoder still runs once
when the screen rejects; original samples and metadata remain untouched.

## Implementation and Safety

- `tools/patch_tiles.rs`: compact fingerprint/origin index, exact comparison,
  grouping, raw-coverage priority and constant-time-per-tile atlas placement.
- Hash equality is not proof: every retained repeat is compared across all
  participating planes. Collisions are exactly sorted/grouped, not discarded
  or walked through an unbounded quadratic collision chain.
- Uniform tiles are skipped; ordinary prediction/RLE already handles them.
  Partial right/bottom tiles are untouched. This detector is not a superset of
  the text/component detector: unaligned occurrences can be missed.
- `tools/patches.rs` shares validated exact Replace plan construction; it
  checks matching original samples before clearing the residual.
- `patch_frame.rs` shares metadata/channel assembly. Full-resolution integer
  1..16-bit extras, alpha and RGB under zero alpha participate in exact matching.
  Other extras keep original residual samples and None blending. Wide/floating
  main color declines discovery; no samples are reinterpreted or rounded.
- Shared encoding emits a compact ReferenceOnly atlas in slot 3 and an ordinary
  visible frame with standard Patch Dictionary Replace entries. No blending
  arithmetic or alpha normalization is introduced.
- Slot reservations, configured-detector conflicts, frame-streaming behavior,
  indexed-output rejection and strict region-streaming rejection use the same
  central `EncodeOptions::has_patches()` predicate.

No new unsafe code, foreign-language runtime or SIMD dependency is introduced.
Mixed wide/subsampled/floating decoder combinations retain their existing
limitations; passing through those planes is not a new decoder support claim.

## Reproduce

```powershell
cargo build --release -p rs-jxl-cli --example patch_lab
target/release/examples/patch_lab input.png --output trial.jxl --effort 9 --threads 8 --mode tiles
python tools/benchmark_tile_patches.py --encoder target/release/examples/patch_lab.exe --decoder PATH_TO_OFFICIAL_DJXL --case input.png --effort 9 --threads 8 --repeats 3 --out out/patch-results.json
```

Development-only modes: `off`, `source`, `components`, `tiles`, `guarded`.
`--tile-admission screened` is the default; `force` measures the unfiltered route.
`--tile-size` changes the tile experiment only. Efforts 4..10 are supported by
the lab; `source` respects the existing source detector gate. Its
`source_detector_enabled` flag reports that gate, not proof a dictionary was
emitted. Source/component mode names identify configured candidates; tile
receipts additionally establish whether a concrete plan was produced.

`guarded` is excluded from harness defaults. Explicitly selecting it returns
the smaller of off/tiles and reports the sum of
both complete encode times. It propagates encoding errors. There is no silent
exception fallback, output overwrite or encoded-JXL equality requirement.

The harness rotates execution order, checks original PNG samples with official
djxl, reports per-image medians, bpp and MPix/s, and deletes temporary JXL/PNG
outputs. Encoded streams are temporary only for the independent oracle. Native
timing includes discovery, owned pool setup, reference/dictionary and residual
encoding; excludes process startup, image I/O/decode and oracle verification.
Measurements use a fixed native candidate, not the outer scheduler tournament.

## Forced Discovery Measurements

This first phase predates admission and intentionally forced every detected
dictionary to measure both wins and losses. Reproduce this phase with
`--tile-admission force`; its costs are not the screened route's costs.

Windows x86_64, release profile, 8 encoder threads, effort 9. The independent
oracle is `djxl v0.12.0 a1ebdf2` (Clang 17.0.6). Fifteen real images were selected
before running the experiment: paper/editor/web screenshots, photographs,
illustration, game artwork, manga, RGBA cursor and a 16-bit displacement texture.
Two repeats per mode, rotated execution order; medians below are the midpoint of
two measurements, not confidence intervals. Small time differences are noise.

The real corpus contains 26,872,633 pixels. Report paths under ignored `out/`:

- `tile-patches-20261109/cases.json`: preselected real manifest.
- `tile-patches-20261109/real-e9-tile16.json`: 120 independently decoded trials.
- `tile-patches-20261109/synthetic-e9.json`: 55 trials, including guarded mode.
- `tile-patches-20261109/real-e9-tile8.json`: 24 additional trials.
- `tile-patches-20261109/real-e9-tile32.json`: 24 additional trials.

The forced-phase release executable SHA256 was
`0208d32f3c06ec999883df6622e36986e3ac16b1f9f08df21f7cb025bcd6f00c`.

| Mode | Total bytes | Sum of median seconds | bpp | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Off | 24,124,631 | 16.565 | 7.1819 | 1.622 |
| Source Add | 23,915,273 | 34.174 | 7.1196 | 0.786 |
| Component Replace | 23,973,746 | 34.654 | 7.1370 | 0.775 |
| Exact tiles 16 | 24,040,628 | 32.296 | 7.1569 | 0.832 |

| Image | Off bytes | Source bytes | Component bytes | Tile16 bytes | Off seconds | Tile16 seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper 9bde | 412,539 | 283,434 | 333,037 | 412,539 | 1.534 | 1.502 |
| Editor 3dd | 126,447 | 106,886 | 104,458 | 111,184 | 0.528 | 1.760 |
| wallhaven-49yekn | 707,442 | 707,442 | 707,442 | 679,595 | 0.965 | 3.492 |
| wallhaven-mlzoy1 | 14,178,604 | 14,178,604 | 14,178,604 | 14,178,604 | 5.038 | 5.362 |
| Linen16 | 1,841,476 | 1,841,476 | 1,841,476 | 1,841,476 | 0.301 | 0.334 |
| Cursor RGBA | 14,678 | 14,678 | 14,678 | 14,678 | 0.181 | 0.194 |
| Kenney Preview | 57,072 | 57,072 | 57,072 | 54,049 | 0.172 | 0.716 |
| Manga Aisazu | 1,372,611 | 1,372,611 | 1,372,611 | 1,372,611 | 0.845 | 0.851 |
| DIV2K 0089 | 3,152,231 | 3,152,231 | 3,152,231 | 3,152,231 | 1.861 | 1.920 |
| Set14 monarch | 407,506 | 407,506 | 407,506 | 407,506 | 0.836 | 0.821 |
| Urban100 061 | 958,068 | 958,068 | 958,068 | 958,068 | 0.746 | 0.630 |
| Web 8279 | 36,014 | 34,655 | 35,412 | 37,231 | 0.529 | 0.496 |
| Paper 1eb | 452,765 | 443,978 | 447,535 | 450,116 | 1.430 | 7.844 |
| Screenshot 7dba | 13,910 | 9,519 | 10,246 | 12,923 | 0.184 | 0.430 |
| image.png | 393,268 | 347,113 | 353,370 | 357,817 | 1.414 | 5.944 |

Tile16 beats off on six images, ties on eight, and grows Web8279 by 3.38%.
It introduces two wins absent from both older detectors: wallhaven-49yekn
(-3.94%) and Kenney Preview (-5.30%). It is not a stronger replacement on every
text screenshot. Paper 9bde has no eligible repeated 16x16 grid tile even though
the source text/component detector is effective.

The nine synthetic repeat cases (all eight 8/16-bit Gray/GA/RGB/RGBA combinations
plus a three-pattern RGB grid) shrink by 16.30%..25.71%. Source/component modes
do not reduce those particular fixtures. Smooth and noise controls are unchanged.
These are mechanism tests, not representative corpus compression estimates.

### Tile Size Follow-up

Six cases were selected after the first measurements for an 8/32-pixel tile
comparison: paper9bde, editor3dd, wallhaven-49yekn, Kenney Preview, Web8279 and
paper1eb. Each variant ran twice with off as a rotated control. This is a
post-hoc diagnostic subset, not additional independent corpus evidence.

- Tile8 discovers repeats in paper9bde: 358,078 bytes / 6.924s. The old source
  text detector is still smaller at 283,434 bytes; adding more tiles is not
  automatically a stronger detector.
- Tile32 gives paper1eb 447,440 bytes / 8.496s. It still spends much more time
  than off for a small improvement.
- Web8279 grows with both sizes: 37,192 bytes at tile8 and 36,563 at tile32,
  versus 36,014 off. Changing grid size alone does not fix admission.

### Why Some Candidates Cost More

Tile16 receipt preparation times in the first real pass range from 0.00058 to
0.03380 seconds; they are not the multi-second bottleneck. In particular:

- Paper1eb: 28 groups / 805 positions, receipt 0.0338s; full encode 7.844s
  versus 1.430s off, for only 0.585% less output.
- wallhaven-49yekn: 6 groups / 37 positions, receipt 0.0201s; full encode
  3.492s versus 0.965s off, for 3.94% less output.

`modular/orchestration.rs::groupwise_decision_geometry` explicitly falls back
from source frame-area encoding to full-frame work when a real patch dictionary
exists. Reference encoding, altered residual statistics and changed MA/palette
work are also included. This code explains an important route change; the
receipt alone does not apportion all backend costs. Calling the entire slowdown
"slow tile detection" would be incorrect.

The new detector therefore stays opt-in. No default scheduler change is justified
by these results, and dual-encode guarded mode is not a free no-regression policy.

## Screened Admission Measurements

The user correction was applied before this phase: reject an unpromising route
at discovery time, not after running two complete encodes. The native estimator
and frame-area hurdle described above are the default for the opt-in detector.
There is no hidden full encode inside the estimator.

The same 15 real inputs ran at e9 / 8 threads, twice per mode with rotated
off/tiles order. A further 11 synthetic inputs ran once per mode. Reports:

- `tile-patches-20261109/real-e9-screened.json`: 60 exact pixel checks.
- `tile-patches-20261109/synthetic-e9-screened.json`: 22 exact pixel checks.

Built with rustc `1.97.1 (8bab26f4f 2026-07-14)`, release executable SHA256
`4b02096c88faf9c5bdbd92a38809ce59c589431979ab018e415bc3ff9473540e`.
Both phases use the same official decoder SHA256
`c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.

| Real corpus mode | Total bytes | Sum of median seconds | bpp | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Off control | 24,124,631 | 16.724 | 7.1819 | 1.607 |
| Screened tiles | 24,124,631 | 17.105 | 7.1819 | 1.571 |

Seven real inputs have repeated eligible tiles. All seven are rejected on
nonpositive estimated net gain, before reference/residual allocation and encoding;
the other eight have no eligible repeated tile. Thus this real set sees no size
change relative to off and approximately 2.28% aggregate timing overhead in this
two-repeat measurement. Detector/admission receipts range from 0.59 to 32.63ms;
those receipts exclude the outer Image-to-plane conversion/validation. Overall
native timings include it. Individual small timing changes are not significant.

Selected cases across phases (forced timings are from the earlier run, not a
paired confidence estimate):

| Image | Forced tile bytes / seconds | Screened bytes / seconds | Current off seconds |
| --- | ---: | ---: | ---: |
| Paper1eb | 450,116 / 7.844 | 452,765 / 1.611 | 1.390 |
| image.png | 357,817 / 5.944 | 393,268 / 1.319 | 1.427 |
| wallhaven-49yekn | 679,595 / 3.492 | 707,442 / 0.973 | 0.965 |
| Web8279 | 37,231 / 0.496 | 36,014 / 0.578 | 0.577 |

This rejects the observed Web8279 growth and avoids the expensive weak routes,
but is deliberately conservative: it also misses all six real size wins from
forced tile16. In particular, editor3dd's 12.07% and image.png's 9.01% byte wins
are lost. They are not reclassified as nonexistent just because the proxy is
negative. Better boundary/residual cost estimates would be needed to recover
those wins selectively; this experiment does not justify default activation.

All nine synthetic repeat cases still pass admission and retain their
16.30%..25.71% size reduction; both nonrepeating controls return to the ordinary
route. This demonstrates a live acceptance path rather than an always-off switch.
Their tiny single-run durations are not representative throughput measurements.

## Focused Validation

- 51 patch-tool unit tests passed, including collision injection, exact hidden
  RGB/alpha comparisons, minimum repeats, partial edges, decoder capacity,
  admission rejection/acceptance/margins and old tools.
- 10 new native integration tests passed: 8/16-bit layouts, decoder threads 1/4,
  encoding threads 1/4, mixed 12/31-bit extras, no-repeat inputs, conflicts,
  reference slot reservation, frame/index/region streaming behavior, rejected
  plans and the stronger frame-area admission hurdle.
- 42 encoder tests passed, 1 pre-existing ignored; 35 streaming tests passed.
- 17 harness tests passed; 1 lab parser test passed.
- Workspace/all-target check passed. Clippy passed with the existing
  `needless_borrow` warnings in unrelated fast_context code temporarily exempted.
- All 305 official djxl comparisons above passed exact original PNG samples:
  223 forced-phase/size-follow-up trials plus 82 screened-phase trials.
  One existing incorrect-sRGB-profile warning occurred; pixel checks still passed.

This is Windows evidence, not a Linux runtime claim or full-corpus acceptance.
No encoded-byte identity is used as a compression target. Binary/source hashes
in reports identify measured artifacts, not desired encoder output.
