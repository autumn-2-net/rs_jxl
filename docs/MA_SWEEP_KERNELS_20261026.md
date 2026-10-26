# MA Small-Histogram Sweep Kernels

Date: 2026-10-26. Baseline: `058128cad0aee250ee3cb654a290f092242e0c5a`.
This is backend work in the independent Rust encoder, not a scheduler change.

## Change

`crates/rs-jxl/src/tools/ma_simd.rs` specializes SSE2/AVX2 split sweeps whose
combined weighted population fits the existing contribution table for totals
up to 256. Block counts, tail boundaries and sparse layout are computed once per
sweep. Each threshold moves the same counts and looks up the same f32 terms,
with the original lane accumulation and reduction order.

The learner passes its already-computed node population to select this kernel;
it is not recomputed or inferred by consuming the first threshold. Inconsistent
per-split population totals still
fall back to the general kernel; each histogram row and lookup remains bounded.
Larger sweeps retain the original general loop without an iterator chain or
special first-threshold handling. This is a crate-private scoring interface;
public codec options and APIs are unchanged.
The existing CPU dispatch, AVX512 route and portable scalar fallback are
retained; AVX512 cases already delegated to AVX2 also use this specialization.

256 is a lookup-table acceleration boundary, NOT a sample/leaf/search budget.
No sample is dropped. No predictor, threshold, effort, scheduler candidate,
parallelism setting, resource limit or bitstream syntax is changed.

## Discarded Experiment

An initial change cleared only live rows of sparse, uncached property
histograms. The first three-image trial was 1-2% slower overall per image, with
no measured benefit. That code and its helper were removed. The trial is saved
under `out/kernel-speed-20261026-r2/sparse/`; it is not part of the final code.

## Kernel Diagnostic

An isolated diagnostic builds the baseline and current scoring module into one
process, alternates their order across four rounds, and uses the same counts
and deltas. Each round performs 20,000 sweeps over 15 thresholds. This measures
the kernel only, not an encoder speedup. Artifacts are ignored development output:

- `out/kernel-speed-20261026-r2/sweep_bench.rs`
- `out/kernel-speed-20261026-r2/ma_simd_before.rs`
- `out/kernel-speed-20261026-r2/sweep_bench_v3.csv`

Representative weighted-population-256 measurements:

| Histogram symbols | Before ms | After ms | Kernel speedup |
| ---: | ---: | ---: | ---: |
| 4 | 5.540 | 2.221 | 2.494x |
| 8 | 4.172 | 2.591 | 1.611x |
| 9 | 5.592 | 2.886 | 1.937x |
| 16 | 5.670 | 3.860 | 1.469x |
| 31 | 10.943 | 7.263 | 1.507x |
| 64 | 14.961 | 12.169 | 1.229x |

General-path measurements range from 0.870x to 1.064x in this short diagnostic.
Do not claim a universal speedup or translate these kernel factors directly into
whole-image throughput.

## Measurement Contract

Frozen executables and reports live in `out/kernel-speed-20261026-r2/`.
Final binaries are `known-lab.exe` and `known.exe`; baseline binaries are
`before-lab.exe` and `before.exe`. The `after-*` binaries/reports are the first
scorer revision; `final-*` and `pair-*` record the second revision. Neither is
the final known-population revision.

Windows x86-64, release, effort 9, 16 workers. Every measured image runs the same
full content-pilot matrix from `out/corpus-repair-20261026/pilot.json`:

```text
--search --content-pilot --pilot-parallel-initial --pilot-overlap-automatic
--pilot-near-tie-consensus --subtrees --ma-workers 16 --concat-batches --fast-ranges
```

Primary time includes the entire encoder call, all probes and full candidates;
it excludes CLI startup, source PNG decoding and file I/O. Independent validation
uses official libjxl 0.12 `djxl` and original decoded samples, including hidden
RGB and original sample precision. JPEG validation reconstructs original bytes.
Compressed-JXL hash equality is not an acceptance criterion.

An unrelated project compiled during the first scorer trial (`after-r0`). Its
timing is not a clean before/after comparison. Second-revision measurements use
adjacent before/after/after/before runs per image (`pair-*`), without builds from
this task in parallel. This is a shared workstation, not an isolated benchmark
machine; other background work can still contribute to timing variation.

## Validation

The final known-population revision passed 520 Modular tests (one explicit
microbenchmark ignored) and all 40 scoring tests, including AVX2, AVX512, sparse/unaligned tails,
packed rows, every population through 257, larger weights, changed totals,
callback unwind and invalid row/layout checks.

```text
cargo test --locked -p rs-jxl --lib modular:: -- --test-threads=8 --quiet
cargo test --locked -p rs-jxl --lib tools::ma_simd:: -- --test-threads=8 --quiet
```

The final release CLI and `encoder_lab` build successfully. The pre-existing
unused `Tools::apply` warning in the lab target remains.

## Whole-Image Results

Final revision: `known-{0,1,2}-{before,known}/report.json`. These are single
adjacent measurements, alternating which binary runs first across images, NOT
multi-run medians. Every final output passed official pixel comparison.

| Image | Before s | After s | Time change | After MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floral_frame_PNG270.png | 22.429066 | 23.311339 | +3.93% | 0.774804 | 5100273 | 2.259042 |
| pizza_PNG44037.png | 38.402443 | 36.423408 | -5.15% | 0.352059 | 17818198 | 11.116226 |
| wallhaven-mlzoy1.png | 26.543793 | 26.365229 | -0.67% | 0.314596 | 14178604 | 13.675351 |

Sizes are unchanged. The sum of call times is 87.375302 s before and 86.099976 s
after (-1.46%). Floral is a measured regression in this pass, not a win. The
earlier second-revision ABBA runs were essentially flat on floral (+0.19%) and
3.38% faster on pizza. Their wallhaven baseline varied from 26.17 to 36.15 s;
that comparison is too noisy to attribute the apparent gain to code.

### Diverse Holdout

`holdout-known-before/report.json` and `holdout-known-known/report.json` cover
12 inputs: icons, textures, urban imagery, natural photography, manga,
transparent artwork, screenshots, scanned text and 16-bit patterns. Each binary
runs once per image; all 24 official pixel checks passed.

| Metric | Before | After |
| --- | ---: | ---: |
| Complete encoder-call time | 15.231756 s | 14.372152 s |
| Pixel-weighted throughput | 1.546445 MPix/s | 1.638939 MPix/s |
| Pixels | 23555076 | 23555076 |
| Encoded bytes | 12590955 | 12590955 |
| Pixel-weighted bpp | 4.276260 | 4.276260 |

This pass is 5.64% shorter overall, but individual inputs remain mixed. The
scanned document contributes much of the gain, so this aggregate is not proof
of a stable corpus-wide speedup. The above slow images still do NOT reach
1 MPix/s. No 500-image/full-corpus rerun was made in this round.

### Production CLI Smoke

`cli-known/report.json` verifies the final production CLI against the frozen
baseline with official `djxl`. `flower_cropped.jpg` reconstructs byte-for-byte
from its 166741-byte JXL (1.233291 bpp); `oi1n2c16.png` reconstructs exact 16-bit
samples from its 141-byte JXL. Both checkpoints passed. CLI process wall times
are not mixed into the encoder-call measurements above.

## Status

The retained change accelerates the small-population scoring primitive without
shrinking the search space or changing the bitstream. Whole-image gains are
modest and not uniform; shared-machine timing noise and the floral result
prevent a no-regression claim. Correctness checks passed, but this report is not
a claim that every image is faster or that the remaining encoder hotspots are
resolved. Larger histogram work, collection and the rest of the full-candidate
pipeline remain outside this optimization.

## Interleaved Retest

Date: 2026-10-26. No encoder source changes or rebuilds were made in this retest.
The same frozen binaries were verified against their earlier reports:

```text
before-lab.exe SHA256 353f94eae03841454127bf2b7a46eb71975f1821ef70ddff0eeb31ef7a4bfd5e
known-lab.exe  SHA256 79e0ff8d6866c3eb9a67bfeef2b70b4f5893ef093a9819a1ddbb0cc98fa05683
```

These are executable provenance hashes, not compressed-output acceptance tests.
The current release `encoder_lab.exe` also matches `known-lab.exe`.

Hardware: Ryzen 9 7945HX, 16 cores / 32 logical processors. Effort 9, 16 workers,
the same full content-pilot matrix, and the same encoder-call timing contract.
Each image has one verified warmup per binary and three measured calls per
binary. Measured order is before/known/known/before/before/known for even-indexed
images and its inverse for odd-indexed images; each binary's warmup immediately
precedes its first measured call. All invocations run serially. No trial is
discarded. The ranges below exclude warmups, as do the medians.

Reports: `out/kernel-speed-20261026-retest/`, with one `report.json` in each
`slow-{case}-run-{order}-{binary}` or `holdout-{case}-run-{order}-{binary}` folder.
The 90 reports contain 90 measured outputs and 30 warmup outputs. All 120
official-djxl comparisons against original pixels passed, including hidden RGB
and 16-bit samples. Each image retained its encoded size across both binaries
and all repetitions. Recorded search routes/candidate lists also agree; this
is a diagnostic observation, not a required bitstream-identity constraint.
JPEG reconstruction was not rerun here; its earlier production CLI check remains
the separate evidence for that path.

This was still a shared-machine test. Initial total CPU utilization was about
15% before the benchmark. Process snapshots during testing showed unrelated
Python, browser and other compute activity; no unrelated process was stopped or
modified. There was no concurrent build from this task. Consequently these are
observed timing distributions, not isolated causal estimates or confidence
intervals. In particular, the wallhaven baseline varied substantially.

### Slow Images

Times are seconds. Positive change means slower. Throughput is based on the
per-image median, not the fastest trial.

| Image | Before median [min, max] | After median [min, max] | Time change | Before / after MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floral_frame_PNG270.png | 24.773942 [23.519588, 26.683875] | 25.913829 [25.455958, 27.004388] | +4.60% | 0.729061 / 0.696991 | 5100273 | 2.259042 |
| pizza_PNG44037.png | 39.018551 [38.234164, 43.782410] | 39.738367 [38.766811, 40.242547] | +1.84% | 0.328644 / 0.322691 | 17818198 | 11.116226 |
| wallhaven-mlzoy1.png | 37.022262 [31.749000, 46.910834] | 29.159724 [28.620896, 31.713136] | -21.24% | 0.224038 / 0.284447 | 14178604 | 13.675351 |

The three medians sum to 100.814755 s before and 94.811920 s after (-5.95%).
This apparent aggregate gain is entirely due to wallhaven, whose baseline is
noisy. Floral remains slower by the median; pizza's previous improvement does
not reproduce in this round. None of these images reaches 1 MPix/s.

### Repeated Diverse Holdout

The same 12 images were tested with the same per-image interleaving and warmups.
Times are seconds; the long 16-bit grid filename is abbreviated in the table.

| Image | Before median [min, max] | After median [min, max] | Time change | After MPix/s | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| actions-edit-cut.png | 0.108897 [0.108771, 0.112136] | 0.109693 [0.108062, 0.111671] | +0.73% | 0.037341 | 7.570312 |
| pkf_sign5.png | 0.321947 [0.318685, 0.325247] | 0.319855 [0.313684, 0.325084] | -0.65% | 0.025612 | 7.192383 |
| pk_hlrmud1c.png | 1.150836 [1.124877, 1.157481] | 1.133896 [1.122199, 1.154227] | -1.47% | 0.057797 | 7.114136 |
| img_086.png | 0.587552 [0.571754, 0.597600] | 0.586651 [0.566528, 0.615655] | -0.15% | 1.340546 | 5.384786 |
| prudential-lossless.png | 0.234651 [0.230748, 0.235339] | 0.232867 [0.227455, 0.233588] | -0.76% | 1.424745 | 7.158034 |
| OL_Lunch.png | 1.130319 [1.066944, 1.149268] | 1.116226 [1.087932, 1.137033] | -1.25% | 0.865793 | 4.928872 |
| img_480x360_3x16bit_RGB_grid_triangle...png | 0.400911 [0.369116, 0.410099] | 0.375173 [0.360353, 0.421889] | -6.42% | 0.460587 | 0.012176 |
| 0109.png | 1.178128 [1.124169, 1.259167] | 1.185472 [1.103050, 1.240854] | +0.62% | 2.333450 | 4.178255 |
| pineapple_PNG2750.png | 2.311200 [2.157959, 2.398126] | 2.440558 [2.370897, 2.501396] | +5.60% | 2.517375 | 7.461176 |
| 2026-07-08_22.29.12.png | 6.208053 [6.076584, 6.258545] | 6.195595 [6.062857, 6.710365] | -0.20% | 0.628472 | 5.666161 |
| 5317.png | 2.510939 [2.347049, 3.628603] | 2.727553 [2.367092, 3.020309] | +8.63% | 3.085183 | 1.109275 |
| oi1n2c16.png | 0.031217 [0.028798, 0.031978] | 0.034033 [0.028264, 0.035134] | +9.02% | 0.030088 | 1.101562 |

The sum of per-image median call times is 16.174649 s before and 16.457571 s
after: **1.75% longer**, not a speedup. Pixel-weighted throughput is 1.456296
versus 1.431261 MPix/s. Both encode 23555076 pixels to 12590955 bytes, or
4.276260 bpp. Small sub-second percentage differences should be interpreted
with their absolute durations and overlapping ranges.

Across all 15 images, the sums of medians are 116.989405 s and 111.269492 s
(-4.89%), with throughput 0.536240 versus 0.563806 MPix/s. This mixed aggregate
must not hide the holdout regression or be presented as a stable corpus gain.
The kernel diagnostic remains evidence of a faster small-population primitive;
the end-to-end retest does **not** establish a reliable overall optimization.
No codec changes, automatic revert, commit or push were performed in this retest.

## Next Tree-Building Investigation

Read-only code inspection during the retest confirmed that branch parallelism
already exists; it should not be proposed as if it were absent:

- `ma_train.rs::find_best_split_with_properties` dispatches property work.
- `train_pending_with_layout` can evaluate two ready children with
  `execution::join` before committing their splits. This is gated by work size
  and the subtree dispatch mode, rather than unlimited recursive parallelism.
- The tested lab's `--subtrees` enables balanced, small-subtree scheduling.
  Detached tasks are collected during the upper-tree traversal and run later
  in `ma_parallel::finish`, in largest-record-count-first order. Each detached
  task executes with `execution::serial` and `parallel: false` internally.
- Detached jobs carry sample ranges, seed/predictor state and redundancy masks,
  but do not transfer the parent histogram cache. Current sibling histogram
  reuse also has imbalance and size gates and is skipped in some detach cases.

The next proposed exact-search experiment is a bounded-concurrency **dynamic
subtree frontier**, allowing a completed split to expose independent child jobs
without waiting for the entire initial frontier to be collected. Large remaining
jobs could expose further branches when workers become idle. Owned, disjoint
sample views and subtree-local results are required: simply spawning closures
that share mutable sample columns or an output vector would be incorrect.
Concurrency/scratch backpressure must not become a new leaf or sample limit.

Alongside scheduling, investigate transferring valid child histogram caches
across task boundaries and extending smaller-child collection plus
`H_large = H_parent - H_small` where it costs less than rebuilding. This is an
established tree-learning technique described by
[LightGBM's histogram-subtraction documentation](https://lightgbm.readthedocs.io/en/stable/Features.html#optimization-in-speed-and-memory-usage).
The current MA learner already uses part of this technique; the proposed work
is broader cache lifetime/reuse, not replacing the MA rate objective with a
generic classifier. Do not assume extending it to every node is automatically
faster: account for cache footprint, collection, subtraction and task imbalance.

Coarse-to-fine or quantile split search is a separate, potentially approximate
experiment. The existing `ma_search::SearchOptions` already exposes such tools;
they may change the tree and encoded size even though pixel reconstruction
remains lossless. Keep them separate from the complete-search scheduling test.
Acceptance remains original-pixel equality, valid official decoding, bpp and
complete encoder-call time, not equality with an earlier compressed bitstream.
These are investigation proposals, not implemented or benchmarked new kernels.
