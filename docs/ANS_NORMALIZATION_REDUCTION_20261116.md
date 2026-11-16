# ANS Normalization Reduction

## Scope

Baseline: `0addc02` (MA contiguous histogram traversal), committed and pushed
before this experiment. This change targets pixel-lossless encoding speed;
there is no new candidate, effort policy, context partition, or bitstream syntax.

A fresh eight-thread profile of the 3024x4032 RGB16 `1515.hdr.png` showed large
inclusive scopes in both MA and histogram-cluster refinement. Cluster refinement
accumulated 56.07 seconds across workers, versus 31.35 seconds in MA histogram
scoring. These are overlapping inclusive elapsed scopes, not CPU time or additive
wall-time shares. The instrumented encode took 21.46 seconds.

`entropy/ans_histogram.rs::rebalance_with_scratch` scans eligible bins for the
best increment or decrement on each balancing iteration. Previously, every
comparison recalculated both the new candidate score and the incumbent score.
The incumbent's bin does not change during a scan.

`best_adjustment` now keeps the incumbent index, shifted comparison key, and
raw score. The caller uses that raw score for the stopping condition. This
replaces `2*N-1` score evaluations with `N`, without changing normalization or
clustering decisions. A singleton still needs one evaluation, so a benchmark
must include small histograms rather than assuming every call becomes faster.

Preserved behavior:

- Strict greater/less comparisons retain the first equal candidate.
- Signed arithmetic shifts remain signed; shifted ties do not imply equal raw
  scores. Termination still uses the unshifted score.
- The decrement zero-count guard and its next-entry step size are unchanged.
- Scores are recalculated for each balancing iteration, after the previous
  chosen bin or balancing count has changed.
- No new unsafe, global cache, allocator, input restriction, or format extension.

## Retained Variant

The initial variant caches the raw score as well as the comparison key and
index. It passed 192 entropy tests and official pixel checks, but had mixed
timing on small/low-effort inputs. This is the retained implementation; a
smaller-state variation was subsequently tested and discarded (see below).

| Measurement | Before (s) | Retained variant (s) | Time change |
| --- | ---: | ---: | ---: |
| e9, 8 threads, 5 images, 3 repeats | 27.2164 | 24.8651 | -8.64% |
| HDR alone within that run | 19.4568 | 17.1511 | -11.85% |
| e9, 1 thread, 3 images, initial 2 repeats | 34.6232 | 35.4893 | +2.50% |
| e9, 1 thread, same 3 images, extra repeat | 29.8803 | 30.5892 | +2.37% |
| e5, 8 threads, 3 images, 3 repeats | 1.0653 | 1.0797 | +1.35% |
| e8, 8 threads, 3 images, 3 repeats | 3.5472 | 3.4924 | -1.54% |

Totals sum per-image medians. Combining all three single-thread repetitions
across the changing timing conditions happens to produce a near-flat total
(34.0293 -> 34.0122 s); this is not evidence of a single-thread speedup. Both
separate batches were slower. All observations remain in the journal.

The e9 screenshot initially regressed 6.35%; six additional paired repetitions
left a +1.46% median change across all nine pairs. At e5 the photo was +3.04%;
at e8 the texture was +5.97%. These motivated trying the smaller two-state
reduction instead of reporting only the favorable HDR result.

Detailed e9/eight-thread results from the original three-repeat run:

| Input | Before (s) | After (s) | After MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| wallhaven-mlzoy1 | 4.1344 | 4.0567 | 2.0446 | 13,853,887 | 13.3622 |
| Paper9bde | 1.5658 | 1.6653 | 0.8744 | 354,791 | 1.9492 |
| 1515.hdr | 19.4568 | 17.1511 | 0.7109 | 26,124,323 | 17.1409 |
| DIV2K/0446 | 1.4808 | 1.4290 | 1.9358 | 3,868,622 | 11.1881 |
| rough_linen_Diffuse_1k | 0.5786 | 0.5629 | 1.8664 | 4,643,251 | 35.3561 |

Bytes are unchanged in every timed comparison. The five-image total is
48,844,874 bytes, mean per-image bpp is 15.799299 and pixel-weighted bpp is
15.169121. Aggregate throughput is 0.9465 -> 1.0360 MPix/s. The bpp values above
are per spatial pixel, including all channels; two inputs are 16-bit RGB.

## Reproduction

Runner: `tools/benchmark_ma_kernels.py`, frozen binaries, interleaved variant
order. Native time includes thread-pool setup but excludes input/output,
process startup and decoder verification. Both sides disable the existing
optional MA AVX512 scoring override identically. No competing benchmark or
build runs during timed encodes.

Inputs: `wallhaven-mlzoy1`, Paper9bde, `1515.hdr`, DIV2K/0446 and
`rough_linen_Diffuse_1k`. The smaller control set omits HDR and DIV2K. Source
metadata is normalized by the runner and the snapshot pixels are verified.
Acceptance compares original decoded samples including alpha-zero RGB and
16-bit samples against official libjxl 0.12 `djxl`; encoded JXL byte identity
is not an optimization objective.

Local artifacts are under `../jxl_python/out/ma-followup-20261116/` relative to
this repository. The first variant is frozen in `paired-e9/after.exe`; the
baseline is `paired-e9/before.exe`. Manifests retain executable digests and
source paths, and journals record every timing and pixel-verification result.
The digests identify experimental binaries, not a target output bitstream.

## Rejected Smaller-State Variant

This variant kept only the comparison key and index during the scan, then
recalculated the winner's raw score once. It passed the same 192 tests but did
not demonstrate a better overall tradeoff:

| Measurement | Before (s) | Compact variant (s) | Time change |
| --- | ---: | ---: | ---: |
| e9, 8 threads, 5 images, 3 repeats | 25.9552 | 24.8117 | -4.41% |
| HDR within that run | 18.7464 | 17.4156 | -7.10% |
| e5, 8 threads, 3 images, 3 repeats | 1.0878 | 1.0978 | +0.92% |
| e9, 1 thread, 3 images, 2 repeats | 34.4898 | 35.3521 | +2.50% |

All four non-HDR inputs regressed in the compact e9/eight-thread medians
(+1.47% to +5.56%). Different batches are not a simultaneous comparison of the
two implementations, and their absolute total encode times are very similar.
Nevertheless, the smaller-state hypothesis did not justify retaining a second
change on top of the initial optimization. It has been removed from the code.
Its frozen executable and trials remain in `compact-*` artifact directories.

## Validation And Decision

The independent reduction oracle covers signed extremes, first ties, singleton
inputs, shift collisions and lengths 1 through 256, and checks one score
evaluation per bin. The retained variant passed all 192 entropy-module tests.
Official decoder checks passed for 96 timed outputs of the retained variant
and its baseline, and 60 outputs in the rejected-variant comparison. These are
repeated encodes of five distinct input images, not 156 different images.
Two initial diagnostic baseline outputs were also verified separately.
After restoring the retained implementation, the full Release workspace
libraries, CLI binaries and `portfolio_lab` example rebuilt successfully.
The rebuilt CLI version command and scoped rustfmt check also passed.

Keep the initial raw-score caching implementation locally, without adding
another option or changing search policy. It is useful on the selected heavy
clustering case, not a demonstrated universal speedup. Single-thread and
low-effort regressions remain an explicit qualification. No large corpus,
Linux runtime, or wheel validation was performed in this round. Do not choose
thread/effort gates from this small set alone.
