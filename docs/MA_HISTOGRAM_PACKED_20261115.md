# Packed MA Histogram Accumulation

## Scope

Reduce writes in the existing histogram construction and collection kernels.
No changes to sampling, tree search, effort policy, entropy scoring, cache
admission, decoder syntax, or encoded precision. Inherited-cache subtraction
is unchanged in this experiment.

The baseline is the working tree profiled in `MA_DETAIL_PROFILE_20261115.md`:
`9cba690` plus the preceding serial-reduction and instrumentation changes.
It is not the bare `9cba690` commit. Each benchmark freezes the actual executable
and records its SHA256 in its manifest, for experiment reproducibility only.
Identical encoded JXL bytes are not an acceptance requirement.

## Implementation

### Fresh cache population

`ma_train/cache_packed.rs` handles fresh two-predictor caches. Previously every
record updated six accumulator locations: record count, weight, two token
counts, and two extra-bit counts. The new loop updates four:

- Records and weights share the low/high 32-bit halves of the existing `used`
  storage on 64-bit targets.
- Two extra-bit counts share the low/high halves of the existing first `extra`
  buffer.
- Each predictor still has its own token histogram and stride.

After collection, the counters are unpacked into the existing cache layout.
No additional buffer is allocated, and scoring/subtraction still consume the
original data structure.

The fast path requires at least 512 records, record count and total weight
within `u32`, and each parent's extra-bit total within `u32`. These are dispatch
guards, not new encoding limits: an ineligible node uses the original loop.
The record/weight packing falls back on 32-bit platforms.

### Temporary paired histograms

`ma_histograms::collect_fresh_pair` similarly combines the two extra-bit
accumulators for unremapped adjacent predictors. Four accumulator updates
become three. It requires fresh zeroed extra buffers, at least 512 records,
and both complete extra-bit totals within `u32`. Otherwise it uses the original
separate counters. Compact-token and mapped-property paths remain unchanged.

### Exactness

All contributions are nonnegative. A parent's complete totals bound every
individual property bin, so a low half cannot carry into its high half.
Equality at `u32::MAX` is allowed and tested, including a packed sum equal to
`u64::MAX`. Real `ResidualToken.nbits` values are retained; extra bits are not
inferred from the token number. This matters for the sample storage contract.

The cache and scratch buffers are exclusively owned during accumulation and
unpacked before publication or scoring. Error checks and mutation ordering in
inherited-cache subtraction are not relaxed. In particular, the marginal pass
must still establish constant/duplicate properties before touching potentially
stale predictor tables.

## Measurement

Windows x64, Release build, ordinary lossless encoding, existing runtime SIMD
dispatch with `RS_JXL_MA_AVX512=0`. No outer Portfolio search. Timings are native
encode elapsed time, excluding CLI startup, input/output and oracle verification.
No build or second benchmark ran concurrently with a timed benchmark.

Final aggregates sum per-image median times. Per-image arithmetic mean bpp and
pixel-weighted bpp are distinguished. These are targeted original-resolution
inputs, not a random corpus or a new comparison against official libjxl.
Run-to-run system variation is visible; small differences are not established
universal gains. Detailed profiling is a separate diagnostic run and is not
included in production-throughput tables.

| Image | Dimensions | Samples |
| --- | --- | --- |
| wallhaven-mlzoy1 | 3840x2160 | RGB8 |
| Paper 9bde | 1264x1152 | RGBA8 |
| 1515.hdr | 3024x4032 | RGB16 |
| DIV2K 0446 | 2040x1356 | RGB8 |
| rough_linen_Diffuse_1k | 1024x1026 | RGB16 |

## Detailed Hotspots

One diagnostic run per variant/image, e9, one thread. These are inclusive
elapsed scopes; do not add nested scopes or interpret them as exclusive CPU
time. Kernel attribution is supported by the cache-only intermediate variant.

| Image / scope | Before, s | Cache only, s | Combined, s | Before to combined |
| --- | ---: | ---: | ---: | ---: |
| Photo: cache fused fill | 5.255627 | 4.239538 | 4.255987 | -19.02% |
| Photo: temporary pair fill | 2.119286 | 2.092046 | 1.599589 | -24.52% |
| Photo: cache subtraction | 1.184209 | 1.139778 | 1.152759 | -2.66% |
| Photo: histogram scoring | 1.950608 | 1.906888 | 1.911511 | -2.00% |
| Paper: cache fused fill | 1.082059 | 0.881838 | 0.867930 | -19.79% |
| Paper: temporary pair fill | 0.271673 | 0.270406 | 0.215590 | -20.64% |
| Paper: cache subtraction | 0.190132 | 0.189091 | 0.181235 | -4.68% |
| Paper: histogram scoring | 0.498433 | 0.498982 | 0.491306 | -1.43% |

Subtraction and scoring were not changed; their timing movement is not evidence
of an optimization to those kernels. The two changed fill stages show much
larger improvements. MA learn elapsed time moved from 18.3468 to 16.5564 seconds
on the photo and 4.1093 to 3.8119 seconds on the paper in this diagnostic pass.

## e9 Eight Threads

Three repetitions per image/version, alternating variant order. Full output
sizes and bpp are unchanged on all five inputs.

| Image | Before, s | Combined, s | Time change | Combined MPix/s | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| wallhaven-mlzoy1 | 3.688446 | 3.374325 | -8.52% | 2.4581 | 13.362160 |
| Paper 9bde | 1.366378 | 1.308860 | -4.21% | 1.1125 | 1.949230 |
| 1515.hdr | 16.248447 | 16.143170 | -0.65% | 0.7553 | 17.140864 |
| DIV2K 0446 | 1.298370 | 1.195042 | -7.96% | 2.3148 | 11.188102 |
| rough_linen_Diffuse_1k | 0.507593 | 0.487715 | -3.92% | 2.1542 | 35.356139 |

Total native time: **23.109234 -> 22.509112 seconds (-2.60%)**.
Aggregate throughput: **1.1147 -> 1.1444 MPix/s**.
Total output: **48,844,874 bytes**. Arithmetic mean per-image bpp: **15.799299**;
pixel-weighted bpp: **15.169121**. Mixed 8/16-bit inputs explain the wide bpp
range. The slow HDR input dominates total time and has little measured benefit.

## Ablations

The first two-image cache-only trial (two repeats per version, profiling off)
reduced total time by 5.97% at one thread and 6.83% at eight threads. A separate
two-image temporary-collector ablation measured +1.58% at one thread and -2.34%
at eight threads relative to cache-only. The photo's two one-thread observations
had opposite signs; this pass alone would not establish an incremental gain.

In the five-image/e9/eight-thread matrix, summed medians were 23.109234 seconds
before, 22.867582 cache-only, and 22.509112 combined. Detailed profiling then
confirmed that the temporary collector reduces its intended fill stage, while
cache fill/scoring remain close between the two optimized builds. Do not pool
these differently scoped runs or replace the unfavorable observation.

## e9 Single Thread

Fresh unprofiled before/combined comparison, three repetitions per image,
alternating order. This covers the photo and screenshot, not the full five-image
set; do not compare its aggregate throughput directly to the five-image table.

| Image | Before, s | Combined, s | Time change | Combined MPix/s | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| wallhaven-mlzoy1 | 19.899408 | 18.414314 | -7.46% | 0.4504 | 13.362160 |
| Paper 9bde | 8.041397 | 7.862298 | -2.23% | 0.1852 | 1.949230 |

Total: **27.940805 -> 26.276613 seconds (-5.96%)**;
throughput **0.3490 -> 0.3711 MPix/s**. Output sizes remain unchanged.

## Lower Efforts

The same five inputs, eight threads, two repetitions per variant. These
results show no overall speedup at e5/e8; they are retained rather than replaced
with the more favorable follow-up subset. Output sizes are unchanged throughout.

| Effort | Before, s | Combined, s | Time change | Before MPix/s | Combined MPix/s | Mean bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e5 | 2.273931 | 2.286290 | +0.54% | 11.3285 | 11.2672 | 17.079055 |
| e8 | 14.372634 | 14.428245 | +0.39% | 1.7923 | 1.7854 | 16.832599 |

The e8 texture initially measured 0.554701 -> 0.635736 seconds (+14.61%),
and the e5 photo 0.655745 -> 0.712558 seconds (+8.66%). A five-repeat,
three-variant follow-up investigated both cases:

| Image / effort | Before median, s | Combined median, s | Change |
| --- | ---: | ---: | ---: |
| Photo e5 | 0.709162 | 0.695840 | -1.88% |
| Texture e5 | 0.092897 | 0.093896 | +1.08% |
| Photo e8 | 2.484768 | 2.320645 | -6.61% |
| Texture e8 | 0.555411 | 0.562080 | +1.20% |

The large initial slowdowns did not reproduce at the same magnitude. This does
not prove zero regressions: the e8 texture still has a small positive median
change, and timings are noisy. The original five-image results above remain
the lower-effort aggregate, not the follow-up subset.

A separate one-thread e8 texture detail pass measured cache fill
0.214139 -> 0.171150 seconds and temporary pair fill 0.371630 -> 0.323172 seconds.
Relative to cache-only, temporary pair fill was 0.378158 -> 0.323172 seconds;
cache fill and histogram scoring were nearly unchanged. This supports actual
kernel improvements on this input, but profiling cannot establish the small
unprofiled eight-thread timing difference's cause.

## Decision and Verification

Keep both guarded accumulation paths. They reduce their intended work without
additional storage, altered histogram totals, reduced sampling or new encoding
limits. Do not advertise an across-the-board effort speedup. e9 has a measured
benefit on the targeted photo/document inputs; HDR improvements are small and
lower-effort aggregate timing is effectively flat in this run.

Do not add a speculative subtraction rewrite in this change. The existing
checked removal and constant/duplicate-property ordering remain intact, and
the measured subtraction scopes are much smaller than the two fill scopes on
the primary targets. Future subtraction work needs its own attribution.

- `cargo test -p rs-jxl --lib modular::ma_train:: -- --quiet`: **284 passed**.
- Release `portfolio_lab` build and scoped `rustfmt --check`: passed.
- **201/201 official djxl original-sample comparisons passed**, including all
  timed trials, ablations and separate diagnostic runs across five unique
  originals. This is not a 201-image corpus.
- All per-image output sizes and bpp match their corresponding baseline.
  JXL byte equality was not required or used as the acceptance criterion.
- New tests cover independent weighted scalar references, arbitrary extra-bit
  counts, full 32-bit halves, overflow fallback, both property widths, unequal
  strides, constant/wide bins, reordered/filtered indices and buffer sentinels.
- Validation was local Windows x64. Linux execution and a full corpus run were
  not performed for this targeted change.

## Artifacts and Reproduction

`out/ma-hist-fill-20261115/` contains input and binary manifests, frozen
executables, per-run native receipts, official-decoder verification flags,
timing summaries and separate detail profiles:

- `quick-e9`: initial cache-only ablation, two images, 1/8 threads, two repeats.
- `pair-ablation`: combined vs cache-only, two images, 1/8 threads, two repeats.
- `validation-e9`: three variants, five images, eight threads, three repeats.
- `single-e9`: before vs combined, two images, one thread, three repeats.
- `validation-e8`, `validation-e5`: five images, eight threads, two repeats.
- `low-ablation-e8`, `low-ablation-e5`: three variants, two images, five repeats.
- `detail`: two images, three variants, one thread, one diagnostic repetition.
- `linen-detail-e8`: one image, three variants, 1/8 threads, one diagnostic run.

Example from the repository root, with a fresh output directory:

```powershell
& 'C:/Users/autumn/.conda/envs/py10/python.exe' -u -B tools/benchmark_ma_kernels.py `
  --manifest out/ma-hist-fill-20261115/validation-e9/manifest.json `
  --variant before=out/ma-hist-fill-20261115/quick-e9/before.exe `
  --variant combined=out/ma-hist-fill-20261115/validation-e9/combined.exe `
  --decoder D:/cpp_pj/jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe `
  --out out/ma-hist-fill-repeat --threads 8 --effort 9 --repeats 3 --profile off
```

Each output is decoded and compared to the original samples, including hidden
RGBA values and 16-bit samples, then deleted. Compiled binaries and experiment
artifacts belong under ignored `out/`, not in commits.
