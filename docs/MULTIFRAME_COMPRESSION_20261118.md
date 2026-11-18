# Multiframe Compression, 2026-11-18

Baseline: `f4e240cbbe66d66daf741918bcc1a099955dc51d`, already pushed to
`origin/main` before this round. The new code adds two automatic-search tools
using existing standard JXL layers. It changes neither the codestream syntax
nor the decoder. All older candidates remain available.

## Tools and controls

- `adaptive_splits=true`: propose a two-axis guillotine partition of occupied
  tiles. Prefix/suffix bounding boxes allow narrow columns or rows instead of
  always dividing the longest axis at its midpoint. Measured bytes per pixel
  guide geometry; actual complete encoded bytes select the winner. It requires
  `allow_multi_rect && adaptive_rectangles`, uses the base tile grid and honors
  `max_rectangles`. Cumulative leaf evaluations are bounded to 64 times the
  occupied-tile count; exhausted leaves retain complete coverage. Ties and
  raster output order are deterministic.
- `allow_multiply_mask=true`: clear changed samples with an ordinary Multiply
  layer, then supply their target values with an Add layer. Unchanged samples
  multiply by exactly one and add zero. Both layers use the difference bbox;
  only the second carries the logical duration. Alpha is handled independently,
  including hidden RGB and associated alpha. RGB needs no added alpha channel.
  This is a fixed two-layer operation, independent of the rectangle-count cap.

Both booleans default to true in `AutoOptions`. Python uses the existing
`auto_options` dictionary; its top-level default is now `strategy="auto"` (explicit crop remains available).
CLI switches are `--no-adaptive-splits` and `--no-multiply-mask`. `allow_mask=false`
or `--no-mask` disables both ordinary alpha masks and multiply masks. Reports
include the requested booleans and distinct Split/MultiplyMask strategy labels.

Multiply masks currently require U8/U16 Gray/GA/RGB/RGBA with declared integer
depths 1..16 and at most one full-resolution alpha channel. Color and alpha
depths can differ. Floating/wide data, other extra planes and configured patch
tools use the existing candidates. The patch exclusion is deliberate: generated
Add patches guarantee final integer recovery, not exact intermediate F32 zero/one
mask values. It applies even when a configured detector would find no atlas.

Each mask layer charges its bbox to the submitted-pixel budget, for 2x bbox area
when complete. Pruning reports two planned layers; a discarded trial never
publishes its scratch reference state. Masks bypass the Replace payload cache
and parallel rectangle path. Both crop arrays are materialized together, so this
work counter is not a peak-memory bound. Static `analyze` does not enumerate the
new encoding candidates; its geometry report is not the full encode search.

## Fixed comparison

Windows x86_64, Ryzen 9 7945HX, rustc 1.97.1, release thin LTO, four threads,
effort 9. The same frozen DLLs serve all runs. Production code was frozen before
the build; benchmarks ran after compilation and native tests finished.

`tools/benchmark_multiframe_feedback.py` alternates before/after reusable native
Encoder and Decoder calls, with one warmup and three measured calls per build.
Times include the public Python/FFI calls and output copies. Input loading and
exact comparisons are outside timers. Every call checks original sample words
(including invisible RGB), shape, dtype, frame count, durations and supplied ICC.
Complete output bytes and SHA-256 must remain stable. Speeds use displayed input
pixels, not the larger number of physical layer pixels.

The nine development/control inputs are the existing frozen rate-feedback
manifest, totaling 30 logical frames and 3,804,888 pixels. The photo-edits fixture
has **140 separate 3x3 edits (1,260 changed pixels)**, not the 64 used by the small
synthetic fixtures. It is a derived local-edit experiment, not a real animation.

| Sequence | Complete bytes before -> after | Change | Encode MPix/s before -> after | Decode MPix/s before -> after |
| --- | ---: | ---: | ---: | ---: |
| Revisit first | 450,868 -> 450,868 | 0% | 0.394 -> 0.407 | 11.770 -> 11.727 |
| Revisit middle | 600,822 -> 600,822 | 0% | 0.360 -> 0.345 | 8.661 -> 8.485 |
| Repeated crop | 376,207 -> 376,207 | 0% | 0.354 -> 0.354 | 15.642 -> 15.813 |
| Subtile changes | 164,206 -> 164,206 | 0% | 0.366 -> 0.364 | 30.084 -> 29.749 |
| Flat to textured | 69,601 -> 58,915 | -15.35% | 0.391 -> 0.357 | 13.720 -> 13.100 |
| Textured to flat | 50,594 -> 50,594 | 0% | 0.854 -> 0.863 | 19.293 -> 18.964 |
| Hidden RGB transition | 69,562 -> 59,118 | -15.01% | 0.298 -> 0.288 | 9.773 -> 9.516 |
| Real t3 matching pair | 371,056 -> 371,056 | 0% | 1.108 -> 0.481 | 45.675 -> 48.729 |
| Photo with flat intro and edits | 240,984 -> 165,848 | -31.18% | 1.181 -> 0.995 | 52.504 -> 73.079 |
| **Total / aggregate throughput** | **2,393,900 -> 2,297,634** | **-4.02%** | **0.764 -> 0.574** | **29.814 -> 32.347** |

Aggregate throughput is total logical pixels divided by the sum of per-case
median times, not an average of MPix/s. Encoding throughput falls 24.9%; decoding
throughput rises 8.5% on this set. There is no decoder code change: different
selected layers explain the large photo-edits decode change. Small differences
on identical streams are timing variability.

The unrelated real t3 pair exposes the search cost: a multiply-mask trial loses,
yet encoding time rises about 2.30x. This is a compression-search improvement,
not an overall encoder speed improvement. Disabling a new tool is available
when that cost matters. These measurements establish neither a universal size
gain nor a globally optimal frame/reference plan.

## Previously unused real pairs

Two source pairs were frozen before codec evaluation and were not used to develop
these algorithms in this round. This does not claim they have never appeared
elsewhere in repository development. Each uses original source p0/p1 ordering,
with nominal test durations of one tick; neither is asserted to be a natural
animation. No resize, crop, recolor or EXIF orientation transform was applied.

- t1: two original 4096x4096 RGBA8 PNGs, sRGB intent 0, no ICC.
- t2: two original 2000x2183 RGB8 JPEG pixel decodes, identical original 536-byte ICC.
- t3's unused p0 is 702x1000 and incompatible with the earlier 638x900 pair; it
  was recorded as excluded instead of modifying its geometry.

Together these are 42,286,432 logical pixels. t1 stays at 10,943 bytes; t2 falls
from 2,416,320 to 2,409,055 bytes (-0.30%). t2 selects 32 SplitPrevious rectangles
instead of 32 MergedPrevious rectangles. Total size is 2,427,263 -> 2,419,998
bytes (-0.30%). This separate set confirms a small real-input gain, not the much
larger percentage seen on sparse derived edits.

The separate repeated run uses the same one-warmup/three-measurement protocol:

| Pair | Encode MPix/s before -> after | Decode MPix/s before -> after |
| --- | ---: | ---: |
| t1 | 6.036 -> 5.949 | 100.282 -> 93.358 |
| t2 | 1.114 -> 0.596 | 29.482 -> 32.583 |
| **Aggregate** | **3.157 -> 2.085** | **67.038 -> 67.399** |

The small size gain therefore has a substantial search cost on t2. t1 produces
identical stream bytes in both builds and the decoder is unchanged, so its
timing difference is not evidence of a decoder implementation regression.
`holdout-final.json` records the repeated samples; the initial single-call
`holdout.json` supplied saved streams for the official oracle. Every repeated
stream matches that oracle-checked SHA-256.

## Contribution checks

One-call ablations are for deterministic size attribution, not speed conclusions.
All other options and the original candidates remain identical.

| Sequence | Baseline | Splits only | Multiply masks only | Both |
| --- | ---: | ---: | ---: | ---: |
| Flat to textured | 69,601 | 58,915 | 69,601 | 58,915 |
| Textured to flat | 50,594 | 50,594 | 50,594 | 50,594 |
| Hidden RGB transition | 69,562 | 59,118 | 69,562 | 59,118 |
| Photo with intro and edits | 240,984 | 173,984 | 165,848 | 165,848 |

The photo split proposal reduces covered rectangle area from 258,192 pixels in
16 rectangles to 25,050 pixels in 10 columns. The final winner instead uses two
multiply/add layers, trading a larger scanned bbox for sparse, compressible data.
The small fixtures have patch options configured by the effort projection, so
their production search conservatively excludes multiply masks. Earlier manual
two-layer probes (53,168/53,209 bytes) did not enforce this eligibility restriction;
they are not the shipped automatic-search results.

Disabling **both** new booleans reproduces all nine baseline codestreams byte for
byte, including SHA-256. Rate feedback remains enabled independently.

## Validation and artifacts

- Native focused checks: 45 multiframe unit tests and 61 integration tests,
  including new splits/masks, original auto/pruning, rate feedback, scales,
  retained references/patches, lookahead, parallel trials and metadata references.
  The old merged-specific test explicitly disables new tools to retain its
  intended winner; its targeted official-decoder test also passes.
- Mask tests cover actual winning two-layer candidates, RGB/RGBA8/16, reduced
  gray and unequal alpha depths, associated/hidden samples, First/Retained and
  zero-duration references, a 43-state same-slot chain, exact work limits, and
  identical bytes with one/four threads and pruning on/off. Unsupported data and
  configured patch tools preserve their fallback paths.
- FFI: 31 tests pass; CLI multiframe: 21 pass; Python multiframe: 29 pass with
  two separate opt-in oracle tests skipped. Rust 1.89 workspace check, changed
  Rust formatting and `git diff --check` pass.
- Official `djxl` verifies all original samples for before/after streams from
  all nine development/control and both held-out pairs. Three additional
  boundary fixtures force actual MultiplyMaskPrevious winners with patches off:
  Gray1, RGB12 and RGBA16. Both old/new streams pass official sample comparison.
  Native checks additionally verify durations, metadata and mixed alpha depths.

Artifacts are under `out/multiframe-compression-20261117/` (work began on the
17th). `build-receipt.json` records source hashes and both binaries; `paired.json`
and `holdout-final.json` retain raw timing samples, complete bytes and native
candidate reports. The frozen manifests, `disabled.json`, `splits-only.json`,
`mask-only.json`, `boundary.json`, three `oracle-*/receipt.json` files, logs and
saved streams remain local development artifacts, outside the commit.

Before DLL SHA-256: `229e41646803bc803e748aeccbdbe82821f9a61b36c47160aeaa4c3046a49ae0`.
After DLL SHA-256: `6ab64a8e3ed677eaf2ea4429bb985cdc84b91de9e1b24e92467f6c04915ed599`.

Reproduce the main paired run with a new output path:

```powershell
python tools/benchmark_multiframe_feedback.py --before out/multiframe-compression-20261117/before.dll --after out/multiframe-compression-20261117/after.dll --manifest out/multiframe-rate-20261117/inputs/manifest.json --threads 4 --effort 9 --warmups 1 --repeats 3 --out out/multiframe-compression-20261117/paired-new.json
```

Use the holdout manifest in `holdout-inputs/manifest.json` for its separate run.
For old-policy byte equivalence add `--after-adaptive-splits off
--after-multiply-mask off`; individual switches provide the two ablations.
