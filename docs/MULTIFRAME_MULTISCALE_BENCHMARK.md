# Multiscale Frames and Native Analysis

Date: 2026-10-27. Windows x86-64 release build, Python binding, effort 9,
four native workers. All reported encoded sequences round-trip to their
original native samples, including hidden RGB. No decoder or bitstream changes.

## What Changed

- `AutoOptions.extra_tile_sizes` adds optional rectangle proposals beyond the
  base grid. Exact difference maps, byte-selected winners, safe pruning and
  repeated-rectangle payload reuse remain native Rust.
- `AnalysisOptions.extra_tile_sizes` inspects those proposals without encoding.
  New `geometry` diagnostics distinguish single-bbox area from multiple-layer
  area, total proposed work and potentially reusable identical rectangle work.
- `AnalysisOptions.history_window` optionally compares older logical inputs.
  History entries are opportunities, not assertions about live decoder slots.
- Python and C ABI only validate/marshal controls and project native reports.
  See [parameters](ENCODING_PARAMETERS.md), [analysis](MULTIFRAME_ANALYSIS.md),
  and [automatic frames](MULTIFRAME_AUTO.md).

## Method

`base` below is `strategy="auto"` with retained references and one-input
lookahead, pruning and adaptive rectangles, base tile 32 and max 32 rectangles.
`scales` adds only `extra_tile_sizes=[8,16,64]`. `single` sums independently
encoded complete images. Both search modes use the same rebuilt DLL.
These are single measurements, without warmups/repetitions; timing differences
are observations, not established speedups. No builds/tests ran during timing.
Other applications were not controlled.

Times include Python packing, FFI and codec setup, but exclude input file I/O,
initial image decoding and correctness comparisons. bpp uses all logical
full-canvas pixels; MPix/s does not count candidate pixels as useful throughput.
JPEG inputs are decoded once and every mode receives the same pixels and
supported metadata. This is **not JPEG byte-reversible transcoding**.

## Real Sequences

| Input / mode | Bytes | bpp | Encode s | Encode MPix/s | Decode s | Decode MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t1 / base | 12,478 | 0.001487 | 20.764 | 3.232 | 2.986 | 22.478 |
| t1 / scales | 12,466 | 0.001486 | 22.063 | 3.042 | 3.589 | 18.699 |
| t1 / single | 45,714 | 0.005450 | 26.240 | 2.558 | 1.615 | 41.554 |
| t2 / base | 3,623,903 | 1.660056 | 39.851 | 0.438 | 5.101 | 3.424 |
| t2 / scales | 3,593,726 | 1.646233 | 91.696 | 0.190 | 3.879 | 4.502 |
| t2 / single | 4,098,261 | 1.877353 | 10.821 | 1.614 | 0.592 | 29.500 |
| t3 matching pair / base | 380,878 | 2.653278 | 1.960 | 0.586 | 0.034 | 33.441 |
| t3 matching pair / scales | 380,878 | 2.653278 | 5.975 | 0.192 | 0.037 | 30.727 |
| t3 matching pair / single | 381,504 | 2.657638 | 0.880 | 1.306 | 0.036 | 32.347 |

t1: four 4096x4096 PNG inputs. t2: four 2000x2183 JPEG-derived RGB inputs.
The full t3 folder was correctly rejected: p0 is 702x1000, while p1/p2 are
638x900. No resizing/padding or metadata conversion was performed. The separate
t3 test explicitly selects only `146810648_p1.jpg` and `146810648_p2.jpg`, whose
sample types and metadata also match. Subset input-decode timing was not measured.

- t1 saves 12 bytes (0.096%); no meaningful reason to enable extra search.
- t2 saves 30,177 bytes (0.833%) at 2.30x encoding time. This fails the practical
  speed/size tradeoff for a default setting. Its frame 2 winner uses grid 64;
  frame 3 uses grid 8. Grid 16 is not a byte winner in this sequence.
- t3's matching pair gains no bytes and takes 3.05x as long. Independent images
  are already close in size and substantially faster.

## Capability Fixtures

Four 256x192 RGB frames from a deterministic random seed. `repeated-crop`
changes one 192x128 area; `subtile-changes` changes four separate 3x3 areas,
some sharing the same coarse tile. These isolate geometry/cache behavior and
are not representative natural-image throughput benchmarks.

| Input / mode | Bytes | bpp | Encode s | Encode MPix/s | Decode s | Decode MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repeated-crop / base | 403,989 | 16.438354 | 3.551 | 0.055 | 0.036 | 5.445 |
| repeated-crop / scales | 403,989 | 16.438354 | 6.025 | 0.033 | 0.035 | 5.676 |
| repeated-crop / single | 645,941 | 26.283407 | 2.479 | 0.079 | 0.057 | 3.438 |
| subtile-changes / base | 176,329 | 7.174845 | 4.312 | 0.046 | 0.019 | 10.117 |
| subtile-changes / scales | 162,278 | 6.603109 | 4.176 | 0.047 | 0.017 | 11.450 |
| subtile-changes / single | 645,789 | 26.277222 | 2.467 | 0.080 | 0.052 | 3.817 |

The dispersed-change fixture improves by 7.969% versus base without an observed
time increase in this pass. The repeated-crop fixture gains nothing and costs
1.70x time despite many payload cache hits. More trials are not automatically
an improvement.

## What Analysis Shows

One native analysis invocation uses `rectangle_bytes_per_pixel=1.0` and
`history_window=8`. Static first/previous candidate analysis is not the actual
byte-driven retained-reference trace. Native timing excludes input decoding,
Python packing and transport.

| Input | Best-area pixels, base -> scales | Proposed layers, base -> scales | Analysis ms, base -> scales |
| --- | ---: | ---: | ---: |
| t1 | 17,723,392 -> 17,702,912 | 106 -> 372 | 64.739 -> 85.506 |
| t2 | 9,505,234 -> 9,443,408 | 168 -> 635 | 17.854 -> 51.321 |
| repeated-crop | 122,880 -> 122,880 | 29 -> 164 | 0.677 -> 0.855 |
| subtile-changes | 53,526 -> 49,260 | 19 -> 39 | 0.516 -> 0.703 |

Best-area totals include the first full frame. Proposed layer counts sum every
candidate before pruning, not only each frame's minimum-area candidate.
t2's tiny geometric improvement with much more trial work is a useful warning;
it is not a compressed-byte prediction or proof allowing safe pruning.
On t1's last input, the default analysis finds a 5.35% canvas-area partition
despite the selected single bbox covering 47.27% of the canvas.

History scanning identifies logical frame 1 as a better bbox reference for t2's
frame 3, and finds the middle-state return in `revisit-middle`. Those results
are observations only: an encoder may already have evicted that reference.

## Adoption and Verification

Extra grids remain **off by default**. This round adds a compression tool and
useful diagnostics, not a blanket faster/smaller policy. Avoid broad grid search
for sequences such as these real t2/t3 inputs. Future selection needs better
cost feedback or temporal tools rather than indiscriminately adding candidates.

- 36 targeted native integration tests passed (analysis, scales, pruning,
  lookahead, generated-patch coexistence).
- Two geometry unit tests passed (stable ties, repeated work and overflow),
  together with 13 benchmark-harness tests.
- 82 CLI and 47 FFI tests passed; two pre-existing CLI external-fixture tests
  stayed ignored. These are not cross-platform execution results.
- 174 Python tests passed, including CUDA input handling and official djxl
  0.12/0.13 decoding of small hidden-RGBA 8/16-bit multiscale cases with mixed
  zero/nonzero durations. Real benchmark sequences use exact native decoding.
- Strict all-target Clippy and formatting checks passed. No full-corpus run.

Raw generated JSON/Markdown reports: `out/multiframe-multiscale-e9/real`,
`synthetic`, `t3-matching-pair`; full native analysis: `analysis.json`.
The original `real` report retains its failed status for the incompatible full
t3 input; t1/t2 and the explicit matching pair passed. No failure was hidden.
The prior DLL is retained under `out/multiframe-multiscale-baseline/` but was
not needed for the same-build option comparison.

Reproduce real t1/t2 with `scripts/benchmark_multiframe_native.py`, supplying
the folders, `--native target/release/rs_jxl_ffi.dll --effort 9 --threads 4
--modes auto-lookahead auto-multiscale independent --out <report.json>`.
The synthetic driver is `scripts/benchmark_multiframe_reuse.py` with the same
native/effort/thread/mode controls and `--cases subtile-changes repeated-crop`.
