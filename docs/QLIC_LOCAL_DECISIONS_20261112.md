# QLIC-Inspired Local Decisions

## Scope

Base: `9fa2aefe5de10f64e22848aabb1b6be497bb8f3f`. Two independent,
opt-in native Rust experiments. Ordinary effort policy, JPEG recompression,
the decoder and the Python ABI are unchanged.

Conceptual references are QLIC's [architecture](https://github.com/OzzyMitchell/QLIC/blob/main/docs/architecture.md)
and [predictor mathematics](https://github.com/OzzyMitchell/QLIC/blob/main/docs/predictor-math.md).
This is an independent implementation using existing JXL tools, not a port of
QLIC code, predictors, arithmetic coder, transforms or block commands. The
author's reported corpus results are not measurements of this repository.

## Block-Local Predictor Trees

`ModularOptions.block_context = Some(block_context::Options { tile_size: 16,
..Default::default() })` replaces MA learning, not the rest of the encoding backend.
The requested tile edge is 8 or 16. Standard Zero, Left, Top and Gradient
predictors are scored using signed-residual HybridUint histograms and extra
bits, with estimated leaf/split description costs. A balanced spatial hierarchy
merges regions when splitting is not estimated to pay. It emits only existing
MA channel, stream, y and x tests. Causal neighbors are never reset at a tile
boundary. Channel IDs, stream IDs and metadata planes keep their original meaning.

The plan coarsens under the existing configured leaf ceiling and a separate
configurable `max_plan_leaves` planning-work ceiling (default 4096); it never
rejects an image for having too many requested tiles. This is not exhaustive
rectangular segmentation. The
receipt reports requested tiles, explored-domain coarsening, emitted leaves,
split types and selected predictors. All transformed samples are scored.
Counters accumulate all learner invocations, including transform candidates
that may lose the final comparison; they are not a dump of only the winning tree.
The estimator cannot promise the smallest final file: histogram clustering,
LZ77 and final entropy coding still matter.

This option conflicts with fixed/custom/fast/seed trees, candidate preparation
reuse, source tree search/scoring overrides and leaf refitting. It is not
automatically a new low-effort default or a warm-start learner. Existing effort
settings still choose transforms and the entropy/LZ77 backend.

## Off-Grid Exact Patches

`TilePatchOptions { off_grid: true, ..Default::default() }` adds arbitrary
destination origins to grid-seeded exact square tiles. A rolling two-dimensional
hash uses O(width * tile_size) scan state. Hashes are only candidates: comparison
checks every original plane, including alpha and invisible RGB, before admission.
The compact reference atlas and standard Replace Patch Dictionary are reused.

The seed dictionary still uses complete grid tiles. Therefore this does not find
every unaligned repeated rectangle, and it is not necessarily a superset of the
grid plan: conservative coarse-cell occupancy may omit nearby nonoverlapping
copies. Constant tiles are excluded. Dictionary/atlas limits remain configurable.
`max_discovery_positions` defaults to 262144 and bounds both the grid seed
count before allocation and the collected occurrences. Exhausting this opt-in
guard declines the entire candidate and returns to ordinary encoding, rather
than failing an image or silently selecting just a discovered prefix. This is
not a total process memory cap. The receipt adds `windows_examined` and
`discovery_limited`.

Cheap existing admission runs before atlas/residual encoding. `force=true`
bypasses only that estimate for diagnostics, not bounds or pixel validation.
An accepted estimate is not a guarantee of actual savings. No automatic
double-encode is hidden in this detector; the existing explicit `guarded` lab
mode can compare two complete encodes at their combined cost.

## Reproduction

Development-only `patch_lab` exposes both experiments independently:

```powershell
cargo build --release -p rs-jxl-cli --example patch_lab -j 4
target\release\examples\patch_lab.exe input.png --output offgrid.jxl --effort 9 --threads 8 --mode tiles --off-grid
target\release\examples\patch_lab.exe input.png --output blocks.jxl --effort 9 --threads 8 --mode off --block-size 16
```

Outputs are create-new only. `--off-grid` requires tiles/guarded; `--block-size`
requires off/source. Source experiments can select reference, residual, or both
with `--block-target`. See [Source Patch speed experiments](SOURCE_PATCH_SPEED_20261112.md).
Neither flag is forwarded to the normal installed CLI or binding.
`--block-leaves N` overrides the planning-work ceiling and requires a block size.
`block_plan_leaves` records that request; `coarsened_views` indicates whether
the actual learner domain still needed coarsening. A requested 8x8 grid is not
an uncoarsened 8x8 experiment unless that counter is zero.
Baseline comparisons must use the same effort and backend, not a differently
tuned palette cap. Report original-pixel correctness, final bytes/bpp, and
native encode seconds/MPix/s; do not treat identical JXL bytes as a target.

The benchmark sidecar is `tools/benchmark_qlic_candidates.py`. It uses an
explicit prebuilt executable and official `djxl`, reports regressions, verifies
original 8/16-bit samples, deletes temporary encoded files and writes a new JSON
report. Native timings include planning/detection and full encoding but exclude
startup, image I/O and independent decoder verification.

For example, replay the fixed mixed corpus plus correctness fixtures (use a
fresh report filename):

```powershell
python -B tools/benchmark_qlic_candidates.py --encoder target/release/examples/patch_lab.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --manifest out/ma-leaf-offset-distribution-20261111/sample30.json --synthetic --modes baseline block8 block16 --repeats 2 --out out/qlic-local-20261112/replay45.json
```

For an individual PNG, replace `--manifest ... --synthetic` with `--case input.png`.
Use `--modes baseline source tiles offgrid` for Patch comparisons, and explicitly
add `--tile-admission force` only to diagnose screening misses. `--tile-size`
affects Patch modes; `--block-leaves` affects only block modes. `--timeout`
(default 300 seconds) guards the external benchmark process, not encoder policy.

Packed 1/2/4-bit PNG input uses the existing corpus preparation adapter. The
encoder receives the original packed pixel payload; only the independent oracle
expands indices/gray samples for comparison. Ordinary 16-bit inputs never go
through an 8-bit conversion. The first mixed run stopped at the harness's old
packed-input restriction; that report is retained as `block30.json`, not counted
as a codec failure or silently excluded input.

An initial block prototype merged equal predictors regardless of differing
residual histograms. This was corrected: equal predictors may retain separate
contexts when their estimated saving pays for the split. A regression test
checks both profitable separation and unprofitable identical-distribution
splits. `block30-complete.json` predates that correction; final block results
come from `block45-rate.json`.

## Measurements

Windows x86_64, release, native direct e9, eight encoder threads. This is not
official libjxl e9, the outer candidate scheduler, or the QLIC author's corpus.
Trials ran serially without concurrent builds. Bytes include the complete JXL.
Corpus bpp uses total bytes / total original pixels, not a per-image average.

Reports live in `out/qlic-local-20261112/`:

- `focused20.json`: five real images plus 15 synthetic inputs, six modes,
  three repetitions. Use its Patch results, not its superseded block prototype.
- `block45-rate.json`: the fixed 30-image mixed manifest plus 15 synthetic
  inputs, baseline/block8/block16, two repetitions, corrected context costs.
- `forced5.json`: five real images, baseline/grid/off-grid, one diagnostic
  repetition with cheap admission bypassed. Single-run times are indicative.
- `uncoarsened3.json` and `uncoarsened-limited3.json`: six real images, three
  modes, one repetition, `--block-leaves 1048576`.

All 663 official decoder comparisons in these completed reports passed against
original pixels, including alpha-zero RGB and full 16-bit samples. These are
repeated candidate checks, not 663 different images. Coverage is 35 real images
and 15 synthetic inputs; this does not establish general corpus performance.

Final runner SHA256:
`83ed522e0a816252009f552ad18ec46c98551fb0ef34ab7183d6b4e08c1d6400`.
The earlier Patch report used
`0d4bcd4d66c7ec4a372c1a22c8f9f34a970e5352c7dda5de9572697e54ac619d`;
its Patch algorithm is unchanged by the later block-context correction.
Official 0.12 `djxl` SHA256:
`c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.

### Block Plans

Only the original mixed 30 images, 68,188,326 pixels; sums of per-image median
times and sizes. No synthetic images are included in this table.

| Route | Bytes | Weighted bpp | Native seconds | MPix/s | Smaller / equal / larger |
| --- | ---: | ---: | ---: | ---: | --- |
| Ordinary e9 | 35,390,526 | 4.152092 | 29.587 | 2.305 | - |
| Block8 | 39,124,007 | 4.590112 | 20.118 | 3.389 | 2 / 1 / 27 |
| Block16 | 39,214,146 | 4.600687 | 19.622 | 3.475 | 2 / 1 / 27 |

Block8 saves 32.0% time but adds 10.55% bytes. Block16 saves 33.7% time but adds
10.80% bytes. `TMW_Screenshot_15.png` is a useful exception: 590,849 -> 579,937 B,
0.899 -> 0.578s with Block8. `358colors.png` is the other size win,
1294 -> 1244 B. Some documents are both bigger and slower: `5020.png` grows
from 3,919,577 to 4,926,481 B with Block8, while measured time rises from about
5.43 to 7.20s.

Block16 had **zero coarsened views across the mixed corpus**. Block8 had 90
coarsened learner views across nine images. Enlarging the planning budget made
the six diagnostic images fully uncoarsened, but did not reverse the outcome:
urban `img_001` and DIV2K `0380` retained their Block8/16 sizes; `baby.png`
improved Block8 by only 45 B, `plant_14.png` by 21 B, and `5203.png` by 0 B.
The uncoarsened PDF remained 568,485 / 568,166 B, versus 412,539 B ordinary e9.
This is not merely an artificial planning-limit loss.

This prototype selects only four standard predictors and coordinate partitions,
not the ordinary learner's richer predictor/property combinations. It does not
show that all block-local decisions, QLIC's predictor set, or a guided MA learner
must fail. It shows this concrete substitution is not a compression upgrade.

### Off-Grid Patches

The existing screened policy rejected every real candidate in the focused five.
It found repetitions, but its conservative boundary/model proxy did not admit
them. This cost roughly 15-75 ms of detector work in the first observation.
It is a heuristic with false negatives, not proof that the route cannot save.

Forced diagnostics separate detection from admission:

| Image | Ordinary B | Grid16 B | Off-grid16 B | Ordinary s | Off-grid s | Off-grid bpp | Off-grid MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper 9bde | 412,539 | 412,539 | 367,951 | 1.215 | 4.120 | 2.021531 | 0.353 |
| Screenshot 7dba | 13,910 | 12,923 | 15,153 | 0.119 | 0.295 | 0.235750 | 1.740 |
| Editor 3dd | 126,447 | 111,184 | 113,539 | 0.388 | 1.379 | 0.424933 | 1.550 |
| image.png | 393,268 | 357,817 | 361,064 | 1.196 | 3.781 | 1.830524 | 0.417 |
| wallhaven-49yekn | 707,442 | 679,595 | 686,312 | 0.823 | 3.273 | 2.647809 | 0.634 |

The PDF is a genuine new Grid16 miss recovered by off-grid matching: 10.81%
smaller than ordinary e9. However, existing Source Add already makes 283,434 B
on that image. Off-grid loses to grid on the other four images, including a
regression above ordinary e9 on 7dba. More detected repeats do not guarantee a
better atlas/residual/dictionary combination. Detection itself took 17-90 ms
in these forced trials; most total time is after detection, not hash lookup.

On deliberately shifted RGBA repeats the screened off-grid route works:
66,075 -> 62,447 B at 8 bits (-5.49%), and 132,222 -> 119,991 B at 16 bits
(-9.25%). Fixed-grid matching missed those repeats. These synthetic fixtures
establish mechanism coverage, not a universal screenshot advantage.

## Decision

Keep both experiments explicit and disabled by default. Do not replace ordinary
e9 MA with the four-predictor block plan, and do not route all screenshots through
off-grid patches. The results favor further work on inexpensive representation
selection and selective refinement of richer MA plans, rather than assuming
more exact repeats or a fixed local grid automatically improves final bytes.

Focused checks passed: 56 Patch tests, 16 block-context tests, two lab argument
tests, 16 sidecar tests, and `cargo check --workspace --all-targets -j 4`.
No Linux/macOS runtime or broad malformed-input conformance claim is made.
