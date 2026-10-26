# Codec Optimization Experiments: 2026-10-25

## Scope And Acceptance

Windows x86_64, Ryzen 9 7945HX, normal Cargo Release (thin LTO, one codegen
unit), no target-cpu override. No new unsafe kernels were retained. No C++
runtime dependency, lossy encoder changes, CI changes, WSL, or ARM work.

Original integer pixels, including invisible RGB, remain the lossless pixel
criterion. JPEG reconstruction must reproduce the original JPEG bytes.
Compressed JXL identity is not a requirement. Primitive-equivalence tests
check arithmetic and state transitions, not an optimization objective.

All experiments below passed their correctness checks. That alone does not
justify retaining slower or unhelpful implementations.

## Rejected Decoder Experiments

The starting checkpoint is the previously verified weighted-lookup and JPEG
refinement implementation from [the single-thread report](DECODER_SINGLE_20261024.md).
Its filename is `out/decoder-perf-20261024/rs-jxl-weighted-lookup.exe`.

| Candidate | PNG time change, 1 thread | PNG time change, 8 threads | Decision |
| --- | ---: | ---: | --- |
| Whole-row AVX2/BMI2/LZCNT plus HybridUint precomputation | +0.51% | +2.04% | Removed row dispatch |
| HybridUint alone, three PNG controls | -0.007% | -0.38% | No useful measured gain; removed |
| Once-per-channel ANS/Huffman reader specialization | +0.53% | +0.44% | Removed |

Positive time change is slower. These tiny differences are not proof of a
universal regression, but none establishes a repeatable useful speedup.
No performance claim is made from fewer source-level operations or use of SIMD.

The row experiment selected CPU features once per row and shared the original
integer loop. The HybridUint experiment cached masks and modulo-32 bit-count
biases without growing its 16-byte configuration. The reader experiment
selected borrowed ANS/Huffman adapters before the pixel loop, retaining the
generic LZ77 path. All three production changes were removed; the previous
profiling hooks and previously accepted optimizations remain.

Row and reader comparisons each used ten PNGs and six JPEGs at one/eight
threads, one excluded warmup, three rotating measured rounds, and five decodes
per invocation. Each run performed 96 exact correctness checks. The isolated
HybridUint run used three PNGs and one sequential JPEG, ten decodes per
invocation, and 24 correctness checks.

Times are process wall time divided by in-process repetitions. PNG timed
output is float32 to match official `djxl --disable_output`; separate checks
compare original integer samples. JPEG timing is original-file reconstruction,
not IDCT rendering. Startup, input I/O, allocation and runner costs are included.
Aggregates sum per-image median seconds; they are not averages of speed ratios.

## Rejected MA Experiments

Production already physically partitions MA sample columns. Two experiments
removed subsequent identity-index reads without reducing samples, predictors,
thresholds, leaf limits or search coverage:

1. Range-based collection and cached sibling subtraction, with indexed tests.
2. Pre-sliced/zip traversal in mapped, marginal and paired collection kernels.

Neither passed the performance gate, so both were removed.

| Full e9 search | Baseline seconds | Range seconds | Range bpp |
| --- | ---: | ---: | ---: |
| Tecnick photo | 5.887 | 6.155 | 8.505000 |
| Clover texture/photo | 8.803 | 9.124 | 12.104523 |
| Web screenshot | 4.684 | 4.695 | 0.244308 |
| PDF screenshot | 19.146 | 19.129 | 0.805742 |

The tighter slice candidate still took 6.262 versus 5.998 seconds on Tecnick,
and 19.350 versus 19.383 seconds on the PDF control. Two measured rounds plus
one warmup were enough to reject it, not to establish a small speedup.
The first range test used three measured rounds plus one warmup. Both runs
also checked single-candidate encoding as a control. All measured sizes were
unchanged; every encoded result passed official djxl 0.12 and 0.13 pixel checks:
128 comparisons for the first experiment and 48 for the tighter candidate.

Fresh, separate source instrumentation confirmed MA remains relevant on the
photo: `ma.split_sweep` accumulated 8.443 seconds inside a 6.234-second
eight-thread full search. Web accumulated 0.686 seconds in that scope and
2.386 in shared LZ77 inside a 4.696-second run. These are overlapping inclusive
worker totals, not wall-time shares. Profiling emits substantial diagnostics
and must not supply throughput claims. The old 125-second MA profile is not a
measurement of today's implementation.

## Reusable Decoder Resources

The retained production change is `decode::Decoder`, documented in
[the decoder API](DECODER_STATUS.md#pixel-api). It lets callers reuse the worker
pool across independent complete inputs. CLI pixel and JPEG repetition loops
now use it. The ordinary free functions and the existing incremental decoder
remain compatible. No decoded-result cache or global thread pool was added.

The final checkpoint is `out/codec-perf-20261025/rs-jxl-reusable-context.exe`,
SHA256 `2f3c5495ffe6b569a85e19e7c89ca3a6144163050495ad55a88ca21160b10477`.
It contains none of the rejected row, HybridUint, reader or MA experiments.

`decoder-context/report.json` uses the original sixteen-case matrix, one/eight
threads, three rotating measured rounds, five decodes per process and one
excluded warmup. All 96 original-output comparisons pass.

| Workload, sum of per-image medians | Threads | Before ms | After ms | Official ms |
| --- | ---: | ---: | ---: | ---: |
| Ten PNGs | 1 | 1240.586 | 1239.304 | 1342.074 |
| Ten PNGs | 8 | 311.757 | 308.422 | 254.436 |
| Six JPEG reconstructions | 1 | 1105.739 | 1113.688 | 887.566 |
| Six JPEG reconstructions | 8 | 347.202 | 345.310 | 465.730 |

This is a small resource-lifecycle improvement, not a demonstrated broad
decoder-kernel acceleration. The observed eight-thread PNG reduction is 1.07%;
JPEG reconstruction is 0.54%. Single-thread changes are near noise. Pixel
decoding still trails the official eight-thread aggregate. JPEG results above
are reconstruction, not JPEG-to-pixel rendering; some individual JPEGs remain
slower despite the faster multithreaded aggregate.

Selected eight-thread details from that run:

| Input | Unchanged input bpp | Before ms | After ms | After MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Tecnick photo | 8.5081 | 47.070 | 46.574 | 30.919 |
| Web screenshot | 0.5072 | 35.828 | 34.893 | 55.062 |
| PDF screenshot | 2.3784 | 56.863 | 55.270 | 26.346 |
| Sequential wallhaven-jexkwm.jpg | 1.4847 | 64.948 | 64.170 | 129.258 |

`decoder-context-single/report.json` uses five controls, five measured rounds
and **one decode per process**, with 30 additional exact-output checks. It does
not establish a consistent single-image speedup: before/after speed ratios are
0.980..1.015 at one thread and 0.967..1.017 at eight threads. Startup and input I/O
remain included. Repeated-loop amortization must not be reported as a kernel
improvement, and short-run aggregate changes do not prove universal non-regression.

The short matrix contained approximately 5% slower readings for eight-thread
clover and single-thread yq5e9x. `decoder-context-controls/report.json` repeats
only these two controls with five rotating measured rounds, twenty decodes per
process and one warmup. All twelve additional original-output checks pass:

| Control | Threads | Before ms | After ms | After MPix/s |
| --- | ---: | ---: | ---: | ---: |
| clover.png | 1 | 185.242 | 184.703 | 5.677 |
| clover.png | 8 | 34.546 | 33.265 | 31.522 |
| wallhaven-yq5e9x.jpg | 1 | 43.800 | 44.206 | 55.640 |
| wallhaven-yq5e9x.jpg | 8 | 18.780 | 18.514 | 132.856 |

The approximately 5% slowdowns did not reproduce. The remaining 0.93% slower
single-thread JPEG reading is not called a speedup or proof of universal
non-regression. The three final-context runs total 138 exact-output checks.
The resource API and CLI reuse are retained; all tests, Release build,
formatting, strict Clippy and whitespace checks are complete.

## Remaining Performance Work

This round does not complete codec optimization. No new encoder speedup is
retained. The measured MA split scoring work, rather than its much smaller
parent-histogram collection scope, is the next encoder target. Common pixel
decode still spends most time in the combined weighted-prediction/context/
entropy loop; do not attribute that entire scope to weight computation alone.
New algorithms may choose different valid lossless streams. Acceptance remains
original-output correctness, measured size and end-to-end runtime, not matching
an earlier compressed stream or improving a primitive microbenchmark alone.

## Retained Regression Coverage

- `decode::Decoder` retains caller-owned worker resources, never parser,
  reference, image or error state. Its context tests compare fresh and reused
  decodes over differing dimensions/formats, selected frames, coalescing modes,
  malformed/truncated inputs, resource failures and exact JPEG reconstruction.
  Both one-thread and four-thread contexts are covered.
- `decode.rs` adds 448 predictor/entropy/bit-depth/row-boundary combinations.
- It also adds 240 tree/configuration/LZ77/bit-depth/layout combinations,
  including invisible RGBA values and mixed predictor trees.
- `decoder_hybrid_uint.rs` compiles the actual maintained HybridUint and
  BitReader sources, checking configuration parsing, extreme tokens, modulo-32
  behavior, bit consumption, continuation, truncation and EOF.

The latter integration harness avoids compiling unrelated upstream unit tests
whose `include_bytes!` fixtures are absent from the vendored checkout. Its
small byteorder adapter uses `std`; buffering and error handling are the actual
BitReader implementation. This does not claim a complete upstream test run.

Final focused checks: 88 decoder integration tests, 328 JPEG unit tests and
82 CLI tests passed. The hostile-input subprocess entry point, nine JPEG
external-oracle tests and two CLI external-fixture tests are deliberately
ignored by ordinary discovery; the hostile parent tests execute their worker.
Strict workspace Clippy with all targets and `decoder-profile` passed. Two
new-test defects were fixed before acceptance: basic metadata has no Debug
implementation, and the sample budget must account for padded backend planes.
Neither fix relaxed production limits or changed pixel comparisons.

## Raw Artifacts

All paths below are ignored outputs under `out/codec-perf-20261025/`:

- `decoder-row-hybrid/report.json` and `REPORT.md`.
- `decoder-hybrid-only/report.json` and `REPORT.md`.
- `decoder-readers/report.json` and `REPORT.md`.
- `encoder-ranges.json` and `encoder-slices-gate.json`.
- `decoder-context/report.json` and `REPORT.md`.
- `decoder-context-single/report.json` and `REPORT.md`.
- `decoder-context-controls/report.json` and `REPORT.md`.

Reports retain exact commands, binary identities, per-case bpp, MPix/s,
timings and validation outcomes. Executable checkpoints identify experiments;
they are not release candidates. Timing runs were serial and did not overlap
builds or other benchmarks. No commits or pushes were made.
