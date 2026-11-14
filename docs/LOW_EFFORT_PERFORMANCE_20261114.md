# Low-effort execution costs, 2026-11-14

## Scope

Optimize the existing low-effort routes before moving any e9 tools into them.
No new predictor, MA learning policy, palette order, LZ77 search policy or
refinement default was added to e2-e5. No decoder syntax changed.

The earlier 200-image curve did show a real low-effort gap against cached
cjxl 0.12 timings. It did not show a general new-vs-old Rust slowdown. This
experiment compares old and new Rust executables on the same machine with
interleaved runs; it does not repeat the official encoder or the full corpus.

## Causes and Changes

### e1 was falling back to Modular

The native FastLossless adapter rejected ordinary default integer candidates,
which then used the same Modular tier-8 policy as e2. This explains the
identical e1/e2 sizes in the previous curve. The native FJXL implementation
was already present; this change fixes its dispatch rather than adding an
experimental compressor.

`scheduler/native_fast.rs` now accepts the ordinary eligible e1 single-candidate
route. Explicit tools, unsupported pixel forms, extra planes, animation,
progressive settings and diagnostic collectors retain the Modular fallback.
The outer image header and container still preserve original metadata.
Validation and output/resource limits still apply.

This is a speed/size tradeoff: FJXL is not required to match the compression
ratio of the old mislabeled e2 route. In this sample e1 is about 11 times as
fast, but total size grows 5.533%. Individual growth ranges from 4.81% to
45.68%; the largest relative increase is the small 16-bit synthetic image
(61,464 -> 89,543 bytes).

The normal CLI `--source-effort 1` and Python `effort=1, source_search=False`
use this route. Cumulative `source_search=True` and manually configured raw
`EncodeOptions` are outside this dispatch change. Inspect
`meta["report"]["implementation"]` for `fast_lossless` versus `modular`.

The benchmark previously bypassed the native adapter as well. Ordinary e1
in `portfolio_lab` now calls the public native single-candidate entry before
attaching Modular-only diagnostic collectors. Its JSON records the actual
implementation. There is still exactly one full encode per measurement.

### Fixed Gradient/Weighted trees used generic pixel emission

`modular/tokens_fixed.rs` computes only the property needed by an eligible
fixed tree and looks up the original context number directly. It avoids
constructing all generic MA properties and walking the tree for each pixel.

- Gradient uses the existing clamped prediction and wide routing sum.
- Weighted uses the existing predictor/history implementation, not a new WP.
- Context maps are bounded to 16,384 entries; larger or incompatible trees
  use the existing generic path. This is not a tree-size or encoding limit.
- Offset, multiplier, cross-channel and mixed-predictor trees retain their
  existing paths. Existing single-leaf SIMD kernels still take precedence.

Both the fixed and generic local raster walks use the existing raster-order
Weighted API, eliminating redundant position bookkeeping. Their local state
is not exposed to callers.

### Fixed HybridInt selection scanned literals again

`entropy/config_search.rs` reuses the already validated per-histogram raw
maxima when the fixed configuration has zero LSBs in its symbol. Symbol
mapping is monotonic in that case, so finding the maximum symbol does not
need another full scan. Nonzero-LSB configurations keep the original scan
and validation. No alphabet or token validity check is removed.

## Measurement

Six original-resolution PNGs, 11,840,564 pixels in total:

| Image | Dimensions | Coverage |
| --- | --- | --- |
| monarch.png | 768 x 512 | photograph |
| 9bde6aaf390109f97622d35ff1e900ee.png | 1264 x 1152 | paper screenshot |
| wallhaven-mlzoy1.png | 3840 x 2160 | large photograph |
| YasasiiAkuma.png | 826 x 1170 | manga |
| jam_PNG20.png | 697 x 800 | RGBA material |
| img_480x360_3x16bit_RGB_circles_corner_tr_0016.png | 480 x 360 | RGB16 synthetic |

For each image: e1-e5, one and eight threads, two repeats, both binaries.
This is 240 encodes, not 240 distinct images. Executables run serially with
alternating old/new order. Native timing includes dispatch and thread-pool
setup but excludes input loading, file I/O and correctness checks. No build
or other benchmark ran concurrently with the timed comparison.

Times below sum the per-image median. MPix/s is total pixels divided by
that time. Mean bpp is the arithmetic mean of per-image bpp, not the
pixel-weighted aggregate. Two repeats on six images are a targeted check,
not a corpus-wide confidence interval; tiny timing differences remain noisy.

### Time and Throughput

| Effort | Threads | Old seconds | New seconds | Elapsed reduction | Old MPix/s | New MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e1 | 1 | 2.05646 | 0.18792 | 90.86% | 5.758 | 63.007 |
| e1 | 8 | 0.61549 | 0.05328 | 91.34% | 19.238 | 222.218 |
| e2 | 1 | 2.19092 | 1.38866 | 36.62% | 5.404 | 8.527 |
| e2 | 8 | 0.64444 | 0.51667 | 19.83% | 18.373 | 22.917 |
| e3 | 1 | 2.99915 | 2.06218 | 31.24% | 3.948 | 5.742 |
| e3 | 8 | 0.77656 | 0.59178 | 23.79% | 15.247 | 20.008 |
| e4 | 1 | 4.46419 | 3.96094 | 11.27% | 2.652 | 2.989 |
| e4 | 8 | 1.10202 | 1.00884 | 8.46% | 10.744 | 11.737 |
| e5 | 1 | 7.01672 | 6.72910 | 4.10% | 1.687 | 1.760 |
| e5 | 8 | 1.75578 | 1.67186 | 4.78% | 6.744 | 7.082 |

e2/e3 have the clearest improvements. e4/e5 have smaller aggregate gains,
with some per-image timings slightly worse. Do not claim a speed guarantee
for every image, especially from small e5 differences.

### Size

Both thread counts produced the following totals in this experiment:

| Effort | Old bytes | New bytes | Old mean bpp | New mean bpp |
| --- | ---: | ---: | ---: | ---: |
| e1 | 17,635,117 | 18,610,880 | 7.95062 | 8.69894 |
| e2 | 17,635,117 | 17,635,117 | 7.95062 | 7.95062 |
| e3 | 17,416,116 | 17,416,116 | 7.74688 | 7.74688 |
| e4 | 16,986,831 | 16,986,831 | 7.39686 | 7.39686 |
| e5 | 16,542,788 | 16,542,788 | 6.87554 | 6.87554 |

No e2-e5 per-image size change was observed. Correctness is decoded original
pixels, including hidden RGB and original sample depth, not equality to a
previous compressed file or its hash.

## Instrumentation

Separate instrumented runs cover monarch, the paper screenshot and the
large photograph at e2/e3/e5 and one/eight threads: 18 trials before and 18
after. These runs are diagnostic, not the interleaved headline timing.
Stage times are inclusive and can overlap; they must not be summed.

Single-thread wallhaven-mlzoy1:

| Effort / stage | Before seconds | After seconds |
| --- | ---: | ---: |
| e2 modular.tokenize | 0.52624 | 0.10677 |
| e2 entropy.uint_config_search.fixed_scan | 0.12967 | 0.05752 |
| e3 modular.tokenize | 1.22947 | 0.54002 |
| e3 entropy.uint_config_search.fixed_scan | 0.12392 | 0.05568 |
| e5 modular.tokenize | 1.28479 | 0.95660 |

This identifies the affected work, but unrelated phase variation should
not be attributed to these code changes. The remaining model/token writing
and MA learning work is still present. For example e2/e3 still build 810
entropy models for 135 groups on this large image, including nested tree
and context-map models rather than six complete encodes per pixel group.

## Verification and Evidence

- All 240 comparison outputs pass official djxl pixel verification.
- All 36 instrumented outputs pass the same check.
- A further 16 e8/e9 encodes on monarch and jam pass official verification,
  with no size change. This is a shared-path guard, not a high-effort benchmark.
- Unit coverage includes fixed-tree borders, full-width signed intermediates,
  context clipping, malformed planes, non-eligible tree fallback, Weighted
  history, HybridInt validation and the actual projected e1 admission rules.
- CLI dispatch and FFI tests cover the public route. The FFI test checks
  Gray/GA/RGB/RGBA, u8/u16, hidden color, partial alpha, linear metadata,
  one/two threads, and explicit RCT fallback.
- The release CLI and FFI DLL were rebuilt. Five targeted Python tests pass
  against that DLL, including direct/reusable e1 encoding, explicit tool
  fallback, invisible RGB, array layouts and 31-bit fallback. Twenty effort
  benchmark/statistics tests also pass.

Local evidence, excluded from Git:

- `out/low-effort-20261114/compare6/{summary.json,trials.jsonl}`
- `out/low-effort-20261114/profile-before/{summary.json,trials.jsonl}`
- `out/low-effort-20261114/profile-after/{summary.json,trials.jsonl}`
- `out/low-effort-20261114/shared-guard/{summary.json,trials.jsonl}`

Old binary SHA256:
`eca43492516d6c4f62b93734b228c9f0325e382c49711e4c682c47b9c84efe08`.
New measured example binary SHA256:
`e1e63907edafbcc41c16fa441c724fe2f270629604b743dda87b25c12741f89f`.
These identify the measured executables, not desired compressed output.
Input hashes, commands, receipts, errors and official decode results are
recorded in the JSON evidence.

Runner: `tools/benchmark_low_effort.py`. Supply repeated `--image` and
`--binary NAME=PATH` arguments plus `--decoder` and a fresh `--out` directory.
Defaults are `--efforts 1 2 3 4 5 --threads 1 8 --repeats 2`; `--profile`
enables stage instrumentation. The runner never changes source images and
deletes temporary encoded/decoded files after checking each trial.

The six-image experiment did not include a full-corpus run, official encoder
rerun, e6/e7 timing claim, Linux run, wheel rebuild or e9-tool backport.

## Follow-Up: 200 Images

At the user's request, the unchanged 200-image cohort from the preceding
effort curves was subsequently remeasured at e1-e5 only, with eight threads
and one repeat per image/effort. All 1,000 outputs passed official djxl pixel
verification; no encoding failure occurred. The complete run took 786.87
seconds including loading, verification and evidence recording. The table
below reports only summed native encode time, excluding those costs.

The comparison here is against the immediately preceding Rust run in
`out/effort-refresh200-20261114`, not the older November 12 Rust baseline.
Historical measurements were not rerun concurrently, so small timing changes
remain sensitive to system load.

| Effort | Previous seconds | New seconds | Elapsed reduction | New mean bpp | Total bytes change |
| --- | ---: | ---: | ---: | ---: | ---: |
| e1 | 31.34365 | 2.76214 | 91.19% | 8.68054 | +12.108% |
| e2 | 31.36166 | 26.32165 | 16.07% | 7.88661 | 0% |
| e3 | 38.65690 | 29.93416 | 22.56% | 7.58375 | 0% |
| e4 | 50.82262 | 47.39070 | 6.75% | 7.31840 | 0% |
| e5 | 71.95151 | 67.95666 | 5.55% | 6.90075 | 0% |

All 200 e1 inputs took FastLossless. The e1 size tradeoff is larger on this
cohort than on the six-image sample. The cached official cjxl 0.12 times for
e1-e5 are 2.36370, 15.45801, 22.03427, 41.81302 and 50.35690 seconds, so this
round does not establish performance parity with the official low efforts.

Evidence: `out/low-effort200-20261114/run/summary.json` and `trials.jsonl`.
The complete curves, with per-image arithmetic mean bpp and total MiB, are
`out/low-effort200-20261114/bpp-time-e1-e9.png` and `size-time-e1-e9.png`.
Each has SVG and CSV companions. Only e1-e5 were remeasured; the hollow green
e6-e9 markers reuse the immediately preceding verified run and intentionally
coincide with its blue points. They are not measurements of the new binary.
No error bars are drawn. CSV rows identify cached points and their reports.

`tools/plot_effort_history.py CURRENT --cached PREVIOUS --outdir OUTPUT`
recreates the combined plots without executing any encoder. It validates
both complete reports and rejects mismatched cohorts, baseline identities,
decoders or measurement settings before combining points.

## Follow-Up: Low-Effort Kernel Work

This round keeps the previous low-effort policy. It does not import e9 search
tools, reduce sample counts, shorten LZ search, or optimize against JXL hashes.
The immutable previous executable is the `e1e63907...` binary above.

Changes to the encoding implementation:

- Fixed/streaming HybridUint models reuse validated clustered counts instead
  of scanning every token again. Adaptive models and injected tables retain
  their original paths.
- Fixed `(4,2,0)` literal histograms use a bounded dense counting layout;
  the writer resolves the shared mapping once per stream, not per token.
- ANS packs extra bits and renormalization bits together and stores reverse
  payload chunks without a second per-chunk width vector or token-count-sized
  upfront allocation.
- Fixed Gradient/Weighted loops peel borders and avoid unused neighbor loads.
  Existing stateless SIMD kernels remain in use. This does not enable the
  experimental narrow WP kernel: its per-pixel range checks have not shown
  a reliable speed advantage.
- No-LZ group encoding emits entropy tokens directly, removing the temporary
  literal and command arrays. Refinement paths that need raw tokens keep them.
- Region sample copies dispatch input type/layout once, preserving signed
  Modular planes, float words and subsampled extra channels. MA gathering uses
  the existing raster kernel after validating its full traversal.

### Same-Session Comparison

Six original images, e1-e5, one/eight threads, two repetitions, interleaved
previous/new/official order: 360 encodes. All pass official djxl verification
against original pixels, including hidden RGB and 16-bit samples. The six
images are the same targeted cohort as the earlier small comparison, not a
replacement for the 200-image curves.

This first checkpoint measures the entropy/predictor/direct-token changes,
before the subsequent region-copy, MA gather and single-entry LZ bucket work:

| Effort | Threads | Previous seconds | Checkpoint seconds | Reduction | Official 0.12 seconds | Checkpoint MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e1 | 1 | 0.19466 | 0.20050 | -3.00% | 0.18227 | 59.06 |
| e1 | 8 | 0.05750 | 0.05871 | -2.09% | 0.04726 | 201.69 |
| e2 | 1 | 1.38237 | 0.85350 | 38.26% | 0.81885 | 13.87 |
| e2 | 8 | 0.51822 | 0.38154 | 26.37% | 0.30934 | 31.03 |
| e3 | 1 | 1.94842 | 1.46390 | 24.87% | 1.53597 | 8.09 |
| e3 | 8 | 0.59503 | 0.46652 | 21.60% | 0.44442 | 25.38 |
| e4 | 1 | 3.86232 | 3.35807 | 13.06% | 3.56353 | 3.53 |
| e4 | 8 | 0.98859 | 0.91502 | 7.44% | 0.88685 | 12.94 |
| e5 | 1 | 6.62591 | 6.47479 | 2.28% | 5.25335 | 1.83 |
| e5 | 8 | 1.70966 | 1.71331 | -0.21% | 1.20016 | 6.91 |

Times sum each image's median; MPix/s uses total pixels divided by that sum.
Rust includes pool setup and excludes image IO; official time is calculated
from cjxl's rounded native throughput, excluding CLI startup and IO. These
are point estimates, not confidence intervals. Unchanged e1 is a control;
millisecond differences are not evidence of an e1 kernel improvement/regression.
Per-image sizes and mean bpp were unchanged relative to the previous Rust
binary at every measured effort. This is an observation, not an acceptance rule.

Official cjxl is `v0.12.0 a1ebdf2`, runtime target AVX2, Clang 17.0.6. It was
rerun here rather than copied from the earlier cached curves. The overall
e3/e4 single-thread results improve on this official run, but neither all
images nor all efforts/thread counts do. e5 remains behind; this checkpoint
does not establish general parity.

### Remaining Work Identified by Instrumentation

On wallhaven-mlzoy1 e5 single-thread, a separate instrumented run records
1.208 s gathering, 1.002 s tokenization, 1.170 s LZ parsing (including 0.759 s
fast seed), and 0.295 s training the prepared MA samples. Gathering and actual
token work are much larger than the split search here. These inclusive stages
must not be added to their parents or interpreted as parallel wall time.

The 270 fast-seed calls for 135 groups are two different streams: pixel
payload and serialized tree. They are not duplicate scans of the same tokens.
Likewise 810 entropy-model calls at e2 include nested tree/context-map models,
not six complete pixel encodings per group. Group payloads already run in
parallel. Profiling output uses synchronized logging and is disabled for
headline performance measurements.

Evidence for this checkpoint (local, excluded from Git):
`out/low-effort-kernels-20261114/compare6/summary.json`, `trials.jsonl`, and
`out/low-effort-kernels-20261114/profile/summary.json`. All six instrumented
outputs also passed official pixel verification. The benchmark runner now
accepts `--cjxl NAME=PATH` in addition to repeated native `--binary` arguments.

### Final Kernel Comparison

The final checkpoint adds the region-copy and MA raster-gather improvements.
It also specializes single-entry LZ hash buckets: 16,384 buckets now occupy
64 KiB instead of 128 KiB. The empty sentinel is outside the validated stream
position domain; other bucket sizes retain their existing ring layout. This
does not change search depth, insertion order, spatial rules or window size.
The changes were measured together, so the bucket change alone has no claimed
isolated speedup.

The same six originals were remeasured at e2-e5, one/eight threads, two repeats
and interleaved previous/final/official order. All 288 outputs passed official
djxl pixel verification. Every image's compressed size and bpp stayed unchanged
relative to the previous Rust binary. The correctness criterion remains
decoded sample equality, not compressed-byte identity.

| Effort | Threads | Previous seconds | Final seconds | Reduction | Official 0.12 seconds | Final MPix/s | Final mean bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2 | 1 | 1.30010 | 0.69983 | 46.17% | 0.81956 | 16.92 | 7.95062 |
| e2 | 8 | 0.52311 | 0.25786 | 50.71% | 0.30901 | 45.92 | 7.95062 |
| e3 | 1 | 1.99336 | 1.28581 | 35.50% | 1.51676 | 9.21 | 7.74688 |
| e3 | 8 | 0.59903 | 0.35070 | 41.45% | 0.41745 | 33.76 | 7.74688 |
| e4 | 1 | 3.76489 | 2.99509 | 20.45% | 3.51440 | 3.95 | 7.39686 |
| e4 | 8 | 0.97038 | 0.69702 | 28.17% | 0.82422 | 16.99 | 7.39686 |
| e5 | 1 | 6.46603 | 5.95143 | 7.96% | 5.10037 | 1.99 | 6.87554 |
| e5 | 8 | 1.61267 | 1.43424 | 11.06% | 1.14909 | 8.26 | 6.87554 |

Times use the same sum-of-per-image-medians method as above. On this cohort,
final e2-e4 elapsed time is 14.6-16.6% below the fresh official measurements.
e5 remains 16.7% slower with one thread and 24.8% slower with eight. The e5
compression results also differ: official mean bpp is 6.98801 versus Rust's
6.87554, so equal effort is not an equal-size comparison.

This is a targeted performance check on photographs, screenshot, manga,
transparent material and 16-bit input, not a whole-corpus result. e1's
FastLossless kernel was not changed in this follow-up and was not remeasured
in the final checkpoint. The existing 200-image charts remain unchanged;
these six-image points must not be substituted into those curves.

Final separate wallhaven-mlzoy1 e5 single-thread instrumentation records:

| Stage | Summed seconds |
| --- | ---: |
| MA gathering | 1.04917 |
| Tokenization | 0.93882 |
| LZ parsing | 1.07978 |
| Fast seed (inside LZ parsing) | 0.70285 |
| Prepared MA training | 0.27303 |
| Best-split search (inside training) | 0.17518 |
| Input plane extraction | 0.02014 |
| Entropy model construction | 0.17939 |
| Token writing | 0.17285 |

Instrumented native time is 4.18579 s; it is not the uninstrumented benchmark
time. Inclusive stages overlap. The next e5 targets are gathering, token
generation and matching, not simply adding more split-search threads.
Group parallelism is active. Existing SIMD is retained; this round primarily
removes redundant work and allocations instead of enabling an unmeasured
AVX512 or causally invalid adjacent-pixel WP implementation.

Final verification:

- 307 focused Rust tests passed: entropy 176, MA training/gathering 68,
  tokenization 11, region samples 6 and LZ matchers 46.
- Python bindings: 33 passed, one CUDA opt-in test skipped, against the new
  release DLL. Coverage includes wide integers, floats, extra channels,
  hidden RGB, multiframe output, JPEG reconstruction and concurrent calls.
- Three benchmark-runner tests passed.
- Another 16 e8/e9 shared-path guard encodes passed official verification,
  without size changes; this is not broad high-effort performance evidence.
- Four final instrumented encodes passed official verification.
- Release CLI, FFI and benchmark example built successfully; the optional
  `experimental-lz77` feature also passed `cargo check`.
- Linux, wheel rebuilding and a fresh 200-image run were not performed here.

Final benchmark executable SHA256:
`2ae8a71382b116edfab9a44f3b444dd772256ed58178f2130043bd64dad7bcbc`.
Local evidence, excluded from Git:

- `out/low-effort-kernels-20261114/final6/{summary.json,trials.jsonl}`
- `out/low-effort-kernels-20261114/shared-guard/{summary.json,trials.jsonl}`
- `out/low-effort-kernels-20261114/profile-final/{summary.json,trials.jsonl}`
