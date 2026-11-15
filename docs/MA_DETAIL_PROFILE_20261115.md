# Detailed MA Profiling

## Scope

Fresh runs of the current working tree: `9cba690` plus the diagnostic-gating
and serial-reduction changes documented in `MA_SERIAL_REDUCTION_20261115.md`.
This follow-up adds instrumentation, not a new encoding policy or algorithm.

Three full-resolution inputs, ordinary e8/e9, 1 and 8 threads:

| Label | Source | Dimensions | Samples |
| --- | --- | --- | --- |
| Photo | `test_img/wallhaven-mlzoy1.png` | 3840x2160 | RGB8 |
| Paper | `9bde6aaf390109f97622d35ff1e900ee.png` | 1264x1152 | RGBA8 |
| HDR | `imazen26_hdr/1515.hdr.png` | 3024x4032 | RGB16 |

Windows x64, workspace Release build, existing runtime SIMD dispatch with
`RS_JXL_MA_AVX512=0`, no outer Portfolio search. Native elapsed times exclude
process startup, image I/O and oracle decoding, but include encoder pool setup.
No compilation or second benchmark was run alongside the profile matrix.

Every matrix entry runs once with stage profiling and once with detail
profiling: 24 encodes. Node-size buckets are enabled in both modes. These are
targeted diagnostic samples, not a corpus benchmark or a new performance
comparison against official libjxl.

## Instrumentation

Existing detailed scopes cover property evaluation, histogram allocation and
fill, cache reuse/subtraction, threshold entropy scoring, candidate extraction,
sample gathering, and parallel dispatch/reduction. This change also measures:

- `ma.quantization`: property threshold construction, including both source
  learner calls. This is distinct from collecting `ma.quantiles`.
- `ma.collect_samples`: legacy sampling, if that path is selected.
- `ma.prepare_thresholds` and `ma.prepare_columns_dedup`: legacy sample setup.
- `ma.prepared_convert`: prepared-column validation and residual conversion.
- `ma.predictor_plan`: per-node predictor equivalence and cost preparation.
- `ma.partition_columns`: physical sample-column partitioning.
- `ma.commit_symmetric`: committing a two-sided split, including its partitions.
- `ma.prepare_children` / `ma.child_parent_histograms`: child state and inherited
  parent histogram preparation, without subsequent child searches.
- `ma.ready_children_join`: the optional joined child searches, including waits.
- `ma.choose_split` and `ma.property_diagnostics`: final split selection and
  profiling-only property statistics.

Fine-grained additions require `RS_JXL_PROFILE_DETAIL=1` as well as profiling.
The low-frequency quantization and legacy collection scopes use stage profiling.
No per-pixel or per-threshold timer was added. These inputs use prepared columns,
so the added legacy-path scopes are not exercised by this matrix.

`tools/benchmark_ma_kernels.py --profile-ma-nodes` now enables the existing
record-count buckets explicitly and records that setting in the manifest.
It rejects the flag when `--profile off` is selected.

## Reading The Numbers

Scopes measure **inclusive elapsed time**, not sampled CPU time. Parent/child
times overlap; parallel worker totals can exceed native elapsed time and can
include waiting. Parent edges record nesting/call counts, not exclusive times.

Use the single-thread tables to understand work distribution. Do not add
`ma.learn`, `ma.find_best_split` and their children, or add
`ma.commit_symmetric` to the complete `ma.partition_columns` total. Similarly,
`ma.hist_fill_pair` includes paired-buffer resizing. `modular.refinement`
includes LZ77/model construction and is not exclusively Offset fitting.

Stage and detail runs are sequential point measurements. Their difference
contains clock/cache/system variation as well as instrumentation overhead; a
negative difference is not a profiling speedup. Fine timers and diagnostic
accounting are not production throughput measurements.

## Native Elapsed Time

Stage profiling plus node buckets; each thread setting has one observation.

| Image | Effort | 1 thread, s | 8 threads, s | Speedup | 8-thread MPix/s | bpp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Photo | e8 | 12.8481 | 2.3217 | 5.53x | 3.573 | 13.65781 |
| Paper | e8 | 2.3854 | 0.4099 | 5.82x | 3.553 | 2.41680 |
| HDR | e8 | 58.1597 | 10.3898 | 5.60x | 1.174 | 20.27812 |
| Photo | e9 | 23.1993 | 3.4056 | 6.81x | 2.435 | 13.36216 |
| Paper | e9 | 7.8833 | 1.4429 | 5.46x | 1.009 | 1.94923 |
| HDR | e9 | 119.7400 | 17.4393 | 6.87x | 0.699 | 17.14086 |

## e9 MA Detail

Single-thread, detailed profiling, seconds. Rows are diagnostic landmarks,
not a complete mutually exclusive partition; especially the first two rows
include the lower-level work below them.

| Stage | Photo | Paper | HDR |
| --- | ---: | ---: | ---: |
| MA total | 16.0625 | 4.9894 | 57.7848 |
| Find-best-split total | 11.4835 | 3.8376 | 50.0339 |
| Gather samples | 3.4946 | 0.8885 | 5.5288 |
| Prepared-column conversion/validation | 0.2404 | 0.0634 | 0.3907 |
| Build cached histograms | 5.0012 | 1.2202 | 6.5717 |
| Subtract inherited cached histograms | 1.0130 | 0.2463 | 1.7666 |
| Collect temporary predictor histograms | 2.0980 | 0.5897 | 8.8359 |
| Threshold entropy scoring | 1.6887 | 0.6625 | 20.7989 |
| Partition sample columns | 0.6185 | 0.1260 | 1.2314 |
| Prepare ordinary child state | 0.0381 | 0.0090 | 0.1406 |

The three histogram collection/subtraction rows do not overlap with each
other. Their sums are 8.1122 / 2.0562 / 17.1743 seconds, versus threshold
scoring at 1.6887 / 0.6625 / 20.7989 seconds. This distinction matters:
photos and this screenshot favor histogram work, while HDR also has a large
scoring hotspot.

Of cached histogram construction, actual fused fill takes 4.8841 / 1.2056 /
6.2697 seconds, respectively 97.7% / 98.8% / 95.4%. Allocation/copy cleanup
alone cannot remove most of that cost. This does not establish whether the
fill kernel is bandwidth-, dependency-, or instruction-bound.

For comparison, detailed eight-thread accumulated times are below. They are
not wall-clock shares and cannot be summed into encoder latency.

| e9 stage, accumulated seconds | Photo | Paper | HDR |
| --- | ---: | ---: | ---: |
| MA total | 19.7570 | 5.7705 | 67.4678 |
| Gather | 4.3376 | 1.0361 | 5.9816 |
| Three histogram collection/subtraction scopes | 9.9385 | 2.2665 | 19.4083 |
| Threshold entropy scoring | 2.0279 | 0.6181 | 21.7815 |
| Partition columns | 0.7942 | 0.1296 | 1.4033 |
| Native elapsed, detail mode | 3.5676 | 1.6528 | 16.0529 |

## e8 MA Detail

Single-thread, detailed profiling, seconds; the same nesting cautions apply.

| Stage | Photo | Paper | HDR |
| --- | ---: | ---: | ---: |
| MA total | 10.2644 | 1.1883 | 40.9286 |
| Find-best-split total | 7.4240 | 0.7716 | 36.0222 |
| Gather samples | 2.0734 | 0.3471 | 3.1733 |
| Prepared-column conversion/validation | 0.1152 | 0.0086 | 0.1701 |
| Build cached histograms | 1.7406 | 0.0563 | 2.3719 |
| Subtract inherited cached histograms | 0.4597 | 0.0567 | 0.9094 |
| Collect temporary predictor histograms | 2.0977 | 0.1997 | 7.1716 |
| Threshold entropy scoring | 1.1263 | 0.1180 | 12.4138 |
| Partition sample columns | 0.4632 | 0.0279 | 0.9077 |
| Prepare ordinary child state | 0.0381 | 0.0043 | 0.1283 |

## Node Size

e9, single-thread stage profiling. Percentages divide each bucket's elapsed
time by the sum of node buckets, not by whole-encode time. Record counts mean
distinct training records after sample processing, not image pixels or summed
sample weights. These runs use two configured predictors.

| Records in searched node | Photo | Paper | HDR |
| --- | ---: | ---: | ---: |
| 0..16 | 0.1% | 0.5% | 0.2% |
| 17..64 | 0.9% | 7.3% | 5.3% |
| 65..256 | 1.7% | 14.6% | 18.5% |
| 257..1024 | 3.4% | 11.1% | 25.0% |
| 1025..4096 | 15.0% | 9.2% | 20.7% |
| >4096 | 78.8% | 57.4% | 30.3% |

Find-best-split calls: 19,399 / 18,316 / 228,821. More than half of HDR's
calls have at most 256 records, but those calls account for about 24% of
searched-node time. Optimizing tiny-node dispatch alone is not sufficient.

The node mix changes with effort. In e8, Photo/Paper/HDR have respectively
69,115 / 14,885 / 577,520 split-search calls. Nodes with at most 256 records
consume 8.7% / 31.7% / 47.2% of stage-profiled node time. Small-node kernels
therefore deserve more attention for e8 HDR than for e9 Photo. This is about
the actual work inside those searches, not evidence that queue dispatch alone
consumes those percentages.

## Profiling Overhead

After the matrix, Paper e9 with eight threads was run in three cyclic orders:
off/stages/detail, detail/off/stages, stages/detail/off. Each mode has three
independent encodes, all verified, with node buckets enabled in profiling modes.

| Mode | Individual native seconds | Median seconds |
| --- | --- | ---: |
| Off | 1.4947, 1.6291, 1.2817 | 1.4947 |
| Stages + nodes | 1.4651, 1.4721, 1.4593 | 1.4651 |
| Detail + nodes | 1.8270, 1.6973, 1.5983 | 1.6973 |

Detail's median is 15.9% above stage-only and 13.6% above off in this small
check. Off/stage noise prevents assigning a precise pure timer overhead from
three repetitions. Do not interpret stage profiling's lower median as a speedup
or transfer these percentages to every image. Detailed parent scopes include
the cost of their nested instrumentation; unexplained parent time is not a
measurement of scheduler overhead.

## Next Targets

1. Cached and temporary histogram filling: reduce repeated column scans and
   dependent counter updates, with separate measurements on large and medium
   nodes. Establish the limiting factor before prescribing wider SIMD.
2. HDR threshold scoring: 20.80 single-thread seconds here. Any scoring-kernel
   optimization should also be checked on the much smaller photo/paper scoring
   share, rather than extrapolating one image to the corpus.
3. Gathering/predictor evaluation remains meaningful, but column conversion,
   child-state construction and final split-choice bookkeeping are not primary
   hotspots in these cases.

This does not make MA the only bottleneck: in coarse single-thread e9 runs,
Paper spends 3.15 seconds in LZ77 versus 3.29 seconds in MA; HDR spends 45.86
seconds in histogram-cluster refinement versus 57.01 seconds in MA. Those
remain separate optimization targets.

## Reproduction

Use `tools/benchmark_ma_kernels.py` with an explicit three-case manifest,
`--threads 1 8 --repeats 1 --effort 8` (then 9), and
`--profile stages --profile-ma-nodes` (then `--profile detail`). Supply the
Release `portfolio_lab` executable and an independent official `djxl`.
Each invocation needs a fresh output directory.

Artifacts: `out/ma-detail-20261115/`, including frozen executables and SHA256
identities, input manifests, raw stage/parent-edge logs and pixel-verified
JSONL receipts. Executable identities document provenance, not a requirement
for encoded JXL bytes to match.

`analysis.json`, `native_runs.csv` and `ma_stages.csv` retain the complete
matrix. `export_profiles.py` validates its shape, bucket call counts and
receipt verification before exporting it; `inspect_profiles.py` prints the
major timing scopes. `source.patch` records the working-tree source changes.

## Verification

- Release build passed.
- Benchmark parser/aggregation tests: 2 passed.
- Profiling-disabled node flag fails early with an argument error, before
  creating an output directory.
- 33/33 official djxl original-sample comparisons passed, including hidden
  RGB in RGBA and all 16-bit integer samples. No comparison of lossy rendered
  previews substitutes for those checks.
- Per-image/effort output size is unchanged between profile modes and thread
  counts. JXL byte identity was neither required nor used as the correctness
  target.
- The source inputs and benchmark outputs are not added to version control.
