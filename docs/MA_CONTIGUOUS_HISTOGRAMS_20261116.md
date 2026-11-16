# MA Contiguous Histogram Traversal

## Scope

Performance-only change on top of `25a3d0258f0454b2bf572f2c905d6399a439da2d`.
No new encoding candidate, effort policy, sampling reduction, or bitstream
extension. This experiment targets paired-predictor histogram construction in
the native Rust pixel-lossless encoder, not JPEG coefficient transcoding.

The learner physically partitions sample columns. A node's index list can
therefore describe a contiguous local range, but both paired-histogram loops
previously still loaded each index and indirectly accessed four source columns.

`PredictorPlan` now validates that range once per eligible node and shares the
proof across its properties. Packed cache filling and fresh paired-histogram
collection then zip four bounded source slices in sequential order. Arbitrary
subsets, permutations, and ineligible packed accumulations retain the original
indexed path. There is no new unsafe code or histogram allocation.

Important boundaries:

- Check every index, not just the endpoints; compute the end with checked add.
- Indices remain local to the current sample view, including nested subtree loans.
- Obtain all four same-length slices before writing histogram counters.
- Preserve existing packed-counter overflow checks and wide fallbacks.
- The 512-record threshold only selects an implementation, not an encoding limit.
- JXL byte identity is not an acceptance requirement. Decode to original samples.

## Method

Windows x86-64 Release builds, same compiler and dependencies, frozen before/after
executables. The existing `tools/benchmark_ma_kernels.py` runner alternates variant
order, records native encoding time, and verifies every output with official
libjxl 0.12 `djxl`. Timing includes native thread-pool setup but excludes image I/O,
process startup, and verification. Both variants use `RS_JXL_MA_AVX512=0`.

Three initial images cover RGB8 photography, RGBA8 document content, and RGB16
texture. Two additional inputs cover a 12.2 MP RGB16 HDR image and another RGB8
photograph. Inputs use the runner's verified, metadata-normalized snapshots.
All pixels, including invisible RGB and 16-bit samples, participate in checking.

- e9, one thread: three images, two paired repetitions per variant.
- e9, eight threads: five images, three paired repetitions per variant.
- e5/e8 controls: the initial three images, eight threads, two repetitions.
- Detailed profiles: three images, one thread, one diagnostic pass per variant.

Totals below sum per-image median native times. The third e9 repeat was added
after noisy HDR/DIV2K pairs; all repetitions are retained. These are targeted
observations, not confidence intervals or a replacement for the full corpus.

## Results

| Configuration | Before (s) | After (s) | Time change | Before MPix/s | After MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| e9, 1 thread, 3 images | 30.7706 | 29.5108 | -4.09% | 0.3510 | 0.3660 |
| e9, 8 threads, 5 images | 25.1078 | 24.5441 | -2.25% | 1.0260 | 1.0495 |
| e5, 8 threads, 3 images | 1.0452 | 1.0026 | -4.07% | 10.3342 | 10.7730 |
| e8, 8 threads, 3 images | 3.4919 | 3.3978 | -2.70% | 3.0932 | 3.1789 |

Do not compare the one-thread and eight-thread aggregate throughput as thread
scaling: those rows use different image sets.

Detailed e9/eight-thread results:

| Image | Before (s) | After (s) | Time change | After MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wallhaven-mlzoy1 | 3.7538 | 3.6121 | -3.78% | 2.2963 | 13,853,887 | 13.3622 |
| Paper9bde | 1.7123 | 1.5742 | -8.07% | 0.9250 | 354,791 | 1.9492 |
| rough_linen_Diffuse_1k | 0.5428 | 0.5244 | -3.39% | 2.0036 | 4,643,251 | 35.3561 |
| 1515.hdr | 17.6393 | 17.4553 | -1.04% | 0.6985 | 26,124,323 | 17.1409 |
| DIV2K/0446 | 1.4597 | 1.3781 | -5.58% | 2.0072 | 3,868,622 | 11.1881 |

Sizes are unchanged in these runs. Five-image total: 48,844,874 bytes;
arithmetic mean per-image bpp: 15.799299; pixel-weighted bpp: 15.169121.
The texture is 16-bit RGB, so its bpp is not bounded by 24.

The improvement is not universal. In the e5 controls, Paper9bde rose from
0.1740 to 0.1806 s (+3.78%) and linen from 0.0889 to 0.0930 s (+4.69%).
At e8 linen was effectively unchanged (+0.23%). HDR's paired runs had mixed
directions; a 1.04% median improvement is weak evidence, not a solved hotspot.

## Diagnostic Profiles

For wallhaven-mlzoy1, the single-thread diagnostic pass recorded:

| Scope | Before (s) | After (s) |
| --- | ---: | ---: |
| cache_fused_fill | 4.1916 | 3.3458 |
| hist_fill_pair | 1.5515 | 1.1215 |
| ma.learn | 15.7036 | 13.8390 |

These scopes are inclusive, may overlap, and come from one instrumented pass.
Do not add them or present their percentage changes as whole-encode speedups.
The document profile also improved in unchanged LZ77/gather code, illustrating
run-to-run variation; paired uninstrumented measurements remain the result.

## Validation And Artifacts

- Release workspace libraries, binaries, and `portfolio_lab` built successfully.
- `cargo test -p rs-jxl --lib modular::ma_train:: -- --quiet`: 286 passed.
- Tests cover byte/wide properties, nonzero contiguous starts, nested borrowed
  subtrees, filtered/reversed indices, full packed halves, overflow rejection,
  noncontiguous interior permutations, and independent scalar histograms.
- Official decoder original-sample equality: 72/72 outputs passed (including
  diagnostic and lower-effort runs), across five distinct source images.
- Scoped rustfmt check passed. No large corpus or cross-platform run this round.

Raw local artifacts: `../jxl_python/out/ma-kernels-20261116/` relative to this
repository. `comparison.json` contains the combined results; each run directory
contains `trials.jsonl`, the input manifest, and frozen executable metadata.
The `paired` directory retains both executables. `summarize.py` checks expected
row counts and verification flags before recomputing the combined tables.

## Decision

Keep the validated sequential traversal as a normal internal fast path with the
indexed fallback. It improves the targeted aggregate without adding candidate
search or changing compression choices. This is a modest kernel optimization,
not a justification to expand expensive e10 searches. HDR still deserves its
own detailed profile before the next performance change.
