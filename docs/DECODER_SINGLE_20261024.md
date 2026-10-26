# Single-Thread and Pixel Decoder Follow-Up: 2026-10-24

## Scope

Follow-up to [parallel JPEG reconstruction](DECODER_PARALLEL_20261024.md).
This iteration first optimizes single-thread original-JPEG reconstruction,
then tests ordinary pixel-lossless decoding. Neither encoder search nor the
compressed representation changes. JPEG correctness means original bytes;
pixel correctness means original integer samples, including hidden RGB.

## JPEG Changes

- Borrow the existing per-channel coefficient scratch instead of zeroing and
  copying a second three-channel block. Empty coefficient channels use a zero
  block, never stale scratch.
- A runtime-dispatched native Rust AVX2 kernel transposes JPEG AC coefficients,
  applies checked chroma-from-luma recovery, validates ranges and narrows to
  i16. It preserves DC. All loads/stores have fixed validated bounds. No ARM
  kernels, C++ linkage or external assembly were added.
- The SIMD kernel commits nothing on unsupported/out-of-range input. The
  existing ordered scalar path then retains overflow checks, failure precedence
  and partial-state semantics. Conservative fast-path arithmetic bounds are not
  new format limits. The vendored decoder's unsafe-code prohibition is retained;
  intrinsics live in its existing low-level `jxl_simd` crate.
- Progressive refinement folds correction-tail collection into the coefficient
  validation pass, packs pending corrections word-wise, and avoids copying the
  EOB table on every block. The bounded entropy writer accepts up to 63 bits at
  once; stuffing, pending bits, output limits and scalar error replay remain.

## Measurement

Native Windows x86_64, Ryzen 9 7945HX, normal Cargo Release (thin LTO,
one codegen unit). The retained starting checkpoint is
`out/decoder-perf-20261024/rs-jxl-parallel-final.exe`, SHA256
`4f4f4276a25db97f3961d824e3364af533dce36706a8dcd867a4794a27a30ff8`.

`tools/compare_decoder_checkpoints.py` runs decoders serially in rotating order,
checks correctness before timing, and checks binary/input identities afterward.
Timings include allocation and amortized process startup/input IO, but not
output-file writes. Official djxl reuses its pool between repetitions; Rust
still creates one per decode API call. No concurrent builds run during timings.
Compressed bpp is `8 * bytes / pixels`; aggregate throughput uses total pixels
divided by the sum of median per-image times, not the average of image rates.

`tools/prepare_decoder_inputs.py` prepares official e9 PNG-derived streams from
the existing fixed subset report. It strips metadata without recompressing PNG
pixels, verifies the snapshot's integer samples, enables `keep_invisible`, and
removes temporary snapshots. The comparison harness independently verifies
the decoded original samples; preparing inputs alone is not verification.

## JPEG Results

First checkpoint: `rs-jxl-single-v1.exe`, SHA256
`b32e79b68d12c10bdb0aee83986d14aba49fbfebf77bbf9830a0077861cb7f09`.
Raw report: `out/decoder-perf-20261024/single-v1/report.json`.
One excluded warmup, three measured batches, ten decodes per batch, both
one and eight requested workers. All 36 byte-verification runs passed.

| JPEG | bpp | Before 1T ms | After 1T ms | Official 1T ms | After MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| wallhaven-yqxzqx | 4.3686 | 578.556 | 519.703 | 410.072 | 28.373 |
| wallhaven-9oo8k1 | 0.9593 | 225.902 | 205.837 | 161.090 | 45.848 |
| wallhaven-vpyekp | 0.3020 | 25.713 | 25.187 | 17.050 | 97.573 |
| wallhaven-yq5e9x | 1.7473 | 54.171 | 50.470 | 36.265 | 48.735 |
| wallhaven-jexkwm | 1.4847 | 244.069 | 183.428 | 136.915 | 45.219 |
| wallhaven-3q3z9d | 2.3864 | 237.032 | 183.191 | 149.522 | 34.272 |

Aggregate over 43.672832 MPix, 2.4228 bpp:

| Threads | Before MPix/s | After MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: |
| 1 | 31.984 | 37.397 | 47.944 |
| 8 | 108.859 | 115.614 | 91.047 |

Single-thread aggregate time fell 14.47%, throughput rose 16.93%; it is **not
yet official single-thread parity** (after throughput is 78.0% of official).
Eight-thread aggregate improved 6.21%; two short cases measured 0.6% and 1.3%
slower, so this is not a claim that every workload improved. The absolute gap
there was below 0.34 ms and includes per-call pool/startup cost.

Separate ten-decode profiling (inclusive stage totals, not benchmark medians)
shows jexkwm at 171.4 ms/call: 106.6 ms coefficients and 61.9 ms rewrite.
The progressive 9oo8k1 case remains writer-heavy: 205.6 ms total, 73.3 ms
coefficients, 130.3 ms rewrite. Both remain useful follow-up targets.

## Pixel Experiments and Retained Diagnostics

Three pixel-backend experiments were removed after measurement. None is part
of the retained implementation, and no ordinary-pixel speedup is claimed:

- Single-gradient row-batched entropy decoding followed by in-place prediction.
  It passed reader-state and output comparisons but produced no convincing
  aggregate gain, including explicitly encoded fixed-gradient route probes.
- A weighted-predictor algebra fast path.
- ANS mask selection in place of boolean-selector multiplications.

An attempted weighted-predictor fast path factored three equal error
coefficients when both spatial coefficients were zero. It preserved exact
arithmetic and passed state comparisons, but the ten-image trial measured
1.13% more aggregate single-thread time and no eight-thread gain. It was
removed, including the extra state flag and branch. Directory names were
assigned before measurement: `pixels-final` and `jpeg-final` record this
rejected intermediate binary (`842d4d70...`), not the final retained build.

An intermediate experiment replaced ANS boolean-selector multiplications with
masks. It passed differential tests but did not show a convincing broad speed
gain; it was removed. `pixels-v1/report.json` records that intermediate combined
experiment, not the final accepted implementation and not an isolated ANS gain.

The retained `RS_JXL_PROFILE=1` instrumentation separates pixel backend
processing, output allocation, frame copies and final color/output conversion.
These are inclusive diagnostic times; profiling is disabled in performance
comparisons. No pixel-loop locks or counters were introduced. One 20-decode
photo profile measured 4.656105 seconds total, 4.579083 in the backend (98.35%),
and 0.054985 in frame copies (1.18%). This locates work in the backend; it does
not yet prove that MA traversal or weighted prediction is the main hotspot.

The final retained AVX2-only build is checkpointed as
`out/decoder-perf-20261024/rs-jxl-retained-avx2.exe`. It keeps the JPEG changes
and coarse pixel diagnostics, but restores the original Modular/ANS algorithms.

## Verification

- 324 core JPEG unit tests passed; nine optional external-oracle tests ignored.
- 52 maintained vendor JPEG tests passed with AVX2 enabled and again without
  SIMD features; two native capture SIMD tests passed, including every signed
  CfL factor and rejection without output mutation.
- During the removed experiment, three batched-gradient differential tests passed over ANS/Huffman, multiple
  HybridUint configurations, I16/I32, narrow rows, cross-row/channel LZ77 copies,
  exhausted input, repeat-before-literal and overflow. Full reader state and
  scratch rows are compared with the previous pixelwise loop.
- The removed weighted-predictor experiment's differential test passed for all 32 equal-coefficient
  values, widths 1/2/3/17/64, extreme i32 inputs, evolving predictions/error state,
  and mismatched headers that must retain the general arithmetic. Four ANS tests
  also passed during the intermediate experiment, which was subsequently removed.
- 91 integration tests passed across `decode`, `decode_jpeg`,
  `jpeg_dc_refinement`, `decode_hostile`, `decode_lossless_edges`,
  `decode_backend_layers`, `decode_streaming`, `decode_wide_integer`, and
  `multiframe_auto`. Two standalone/optional tests are ignored: the hostile
  worker is exercised by its isolated parent tests; the external JPEG corpus
  test was not enabled.
- Strict core/CLI Clippy, formatting, diff whitespace checks and six benchmark
  harness tests passed. No WSL, macOS, CI or Python binding changes.

Vendor unit tests used the existing ignored `out/decoder-vendor-validation`
fixture copy, synchronized with maintained source. No upstream image fixtures
or generated streams/binaries were added to Git.

Before the last gradient rollback, 78 directly affected decoder integration
tests, three row-gradient differential tests, strict Clippy and formatting
checks passed again. The additional 13 multiframe-encoder strategy tests had
already passed; that encoder was not modified. All three rejected changes and
their experiment-only tests were then removed and normal Release rebuilt.

## Experimental Corpus Checks

`out/decoder-perf-20261024/accepted/report.json` is also an intermediate report:
the directory was named before the measurements, and its remaining gradient
experiment was ultimately rejected. All 114 original-byte/pixel checks passed,
but correctness alone does not justify retaining a performance change.
Indices 0..9 are the original ten default-e9 PNG-derived streams; 10..15 are
the six JPEG-reconstruction streams; 16..18 are explicitly selected
fixed-gradient route probes. The groups must not be combined into a headline
aggregate. Default PNG aggregate single-thread throughput was 7.028 versus
7.016 MPix/s; eight-thread throughput was 25.388 versus 25.353 MPix/s. These
are effectively unchanged/slightly worse, not a demonstrated acceleration.

Those intermediate comparisons request one/eight workers, use one excluded
warmup plus three measured batches, and decode five times per process. Short
cases are sensitive to startup/pool costs; these are not kernel-only numbers.
PNG timing matches official djxl's float32 no-output path. Separate output
runs verify original integer samples, including invisible RGB and 16-bit data.

Route probes use the same photo, web screenshot and PDF screenshot as the
existing subset, but are encoded by official 0.12 with `-d 0 -e 9 -P 5 -I 0
-E 0 --keep_invisible=1`. These intentionally different streams isolate the
fixed-gradient path; they are not new compression recommendations, do not
change the default-e9 comparison, and are not evidence of a generic pixel
decoder speedup. Commands and source hashes are in
`gradient-inputs-v2/manifest.json`.

## SIMD Audit and AVX-512 Build

The vendor already contains SIMD reversible color transforms for both I16 and
I32, output conversion and other rendering kernels. They should not be
reimplemented just because the whole decoder remains slow. ANS state evolution
and prediction between adjacent pixels also prevent naive whole-row SIMD.

The workspace originally selected only `sse42`, `avx`, and `brotli` features.
Thus the vendor's existing AVX-512 kernels were not compiled into earlier
decoder checkpoints. This machine exposes AVX-512F/BW/VL, among other subsets.
An opt-in `decoder-avx512` feature now forwards to `jxl-decoder/avx512`:

```powershell
cargo build --locked --release -p rs-jxl-cli --features decoder-avx512
```

Library consumers enable `features = ["decoder-avx512"]` on `rs-jxl`.
The original default stays AVX2-capable. The optional build checks CPU/OS
support at runtime and falls back to AVX2/SSE/scalar. Do not use a global
`target-cpu=native` or `+avx512` flag for a portable binary. This switch affects
the decoder, not encoder selection, lossless semantics or the JXL format.
The initial AVX-512 checkpoint still used AVX2 for the new JPEG AC helper;
enabling the broader backend does not widen every helper automatically.

### Existing Kernels Only

`out/decoder-perf-20261024/avx512/report.json` compares the retained AVX2 build
(`291a2a4d...`) with the same code plus the vendor AVX-512 feature
(`fa2b59b5...`, preserved as `rs-jxl-avx512-stock.exe`). All 114 original
pixel/JPEG-byte checks passed. Settings remain one warmup, three measured
batches, five decodes per batch, rotating decoder order.

| Workload | Threads | AVX2 MPix/s | Existing AVX-512 MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Default PNG (4.5110 bpp) | 1 | 7.137 | 7.117 | 6.715 |
| Default PNG (4.5110 bpp) | 8 | 26.554 | 26.671 | 32.094 |
| JPEG reconstruction (2.4228 bpp) | 1 | 35.450 | 35.207 | 46.223 |
| JPEG reconstruction (2.4228 bpp) | 8 | 97.939 | 99.831 | 85.867 |

These results do not demonstrate a broad gain from enabling the existing
kernels. Fixed-gradient probes are separate: 23.785 to 24.655 MPix/s (1T),
48.315 to 49.763 (8T); their route-specific gain must not be presented as a
default-corpus gain. The feature therefore stays opt-in. This comparison also
verifies the final retained AVX2 binary after the three pixel experiment
rollbacks, not just an earlier intermediate build.

## Kernel Implementation Follow-Up

The feature-only result prompted changes to the kernels themselves:

- AVX2 capture uses wrapping interval translation followed by one signed
  comparison. The original accepted ranges, DC handling and no-commit-on-error
  guarantee are unchanged. Pairing output rows into wider stores was measured
  slower and removed; it is not part of the retained AVX2 change.
- AVX-512 capture has a new four-register transpose, two-source permutations,
  compact range masks and direct 16-element narrowing. Its arithmetic remains
  integer-exact; no floating approximation is introduced.
- An I16 RCT candidate used `(a & b) + ((a ^ b) >> 1)` for overflow-safe floor
  averaging. It was subsequently removed: the corrected native microbenchmark
  did not show a benefit. Neither I16 nor I32 RCT arithmetic is changed in the
  final retained source.

Seven JPEG kernel tests passed against an independent i64 oracle and both SIMD
implementations. They include all AC positions, signed half-step rounding,
every output alignment modulo 64 bytes, guarded buffers, full-width invalid
inputs, cancellation at valid bounds, and unchanged output on rejection.
During the removed experiment, twelve RCT tests passed across scalar/SSE/AVX2/AVX-512, including every i16
operand value in a permutation corpus and extreme i32 wrapping cases.

### JPEG Kernel Microbenchmark

The ignored `paired_release_microbenchmark` uses opaque inputs, 128 blocks,
16,384 passes and 21 alternating pairs. The original AVX2 kernel is retained
only in an ignored local fixture at `out/decoder-perf-20261024/capture-baseline`.
Three alternating old/new executable runs were pinned to logical CPU 4
(affinity mask `0x10`); medians of their reported block-time medians are below.
No compiler or other benchmark ran concurrently. Values are ns per 8x8 block,
not whole-image decoding times.

| Kernel | No CfL | With CfL |
| --- | ---: | ---: |
| Original AVX2 | 6.38 | 18.68 |
| Optimized AVX2 | 6.15 | 16.73 |
| New AVX-512 (in optimized executable) | 4.05 | 12.39 |

AVX2's CfL kernel time falls about 10.4%; its non-CfL difference is small.
AVX-512's time is about 34.1%/25.9% lower than optimized AVX2 for these two
synthetic modes. Across the three runs, AVX2 CfL ranged 17.72..19.29 ns before
and 16.36..16.95 ns after. CPU placement caused larger noise in earlier unpinned
trials, so those are not the headline microbenchmark. These figures do not
imply comparable whole-decoder gains: entropy parsing and JPEG rewriting are
still substantial independent costs.

### Default AVX2 End-to-End Check

`out/decoder-perf-20261024/kernels-avx2/report.json` compares the earlier retained
AVX2 build with optimized AVX2 capture plus the subsequently removed I16 RCT
formula. It is an intermediate report, not the final post-rollback binary.
All 96 original
sample/JPEG-byte checks passed. The output binary is `rs-jxl-kernels-avx2.exe`,
SHA256 `b8b65ca5a516d73e01a34d5df2a84d01840e21a13d5fcbf29a8e3eb55886d3b5`.

| Workload | Threads | Before MPix/s | After MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Default PNG (4.5110 bpp) | 1 | 7.149 | 7.107 | 6.663 |
| Default PNG (4.5110 bpp) | 8 | 26.386 | 25.945 | 31.642 |
| JPEG reconstruction (2.4228 bpp) | 1 | 35.834 | 36.336 | 47.207 |
| JPEG reconstruction (2.4228 bpp) | 8 | 105.453 | 104.639 | 87.725 |

All six JPEG single-thread cases improved by 1.2..2.0%, aggregate throughput
by 1.4%. There is no demonstrated PNG aggregate gain or eight-thread JPEG gain.
PNG throughput measured 0.6%/1.7% lower and eight-thread JPEG 0.8% lower; these
results are not relabeled as non-regression. Per-case eight-thread results vary
in both directions. This small kernel change does not reach official 1T JPEG
performance or establish an improvement across every image.

An initial vendor RCT microbenchmark reported AVX2 27.626 to 21.069 ns/vector
and AVX-512 30.940 to 25.108. **Those figures are not accepted as native-kernel
performance.** Disassembly found that its `Descriptor::call` closure failed to
inline, leaving ten helper calls per vector before and seven after, plus stack
traffic. The corrected direct feature-enabled batch functions measured AVX2
0.402 to 0.463 ns/vector (paired 0.869x), AVX-512 0.649 to 0.658 (0.988x).
There is no positive native-kernel evidence, so the RCT candidate and its
experiment-only maintained tests were removed. The corrected test fixture
remains in ignored `out/decoder-vendor-validation` for local reproduction.
Disassembly of all four corrected batch functions confirmed no per-operation
helper calls, with actual repeated vector loads/arithmetic/stores still present;
timer and validation calls are outside those loops. AVX-512's original formula
also benefits from `vpternlogq`, illustrating why source operation counts alone
were insufficient.
This finding does not prove the same overhead in production RCT,
whose dispatch chain is different. It also does not invalidate the integer
correctness tests or the end-to-end measurements above.

Before the RCT rollback, the candidate source also passed 91 decoder integration tests with the optional
AVX-512 feature: exact 8/16/wide integer samples, hidden RGB, raw/composited
layers, streaming, JPEG reconstruction, malformed input and multiframe strategies.
The hostile subprocess entry point and external-path JPEG corpus test remain
explicitly ignored; their respective parent tests and benchmark JPEG originals
were exercised separately.

### Optional AVX-512 End-to-End Check

`out/decoder-perf-20261024/kernels-avx512/report.json` compares the two optimized
variants before the RCT rollback on the same six JPEG streams. All 36 original-byte checks
passed. AVX-512 checkpoint SHA256 is
`68aac10291251b54c025deabf7537772e69aeb9766a2970d9c93762e11a7a84f`.

| Threads | Optimized AVX2 MPix/s | Optimized AVX-512 MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: |
| 1 | 36.585 | 36.036 | 47.106 |
| 8 | 106.205 | 105.108 | 89.557 |

Despite the faster standalone capture kernel, the full AVX-512 configuration
was 1.5%/1.0% slower in aggregate in this run. It changes the vendor's broader
SIMD dispatch as well as JPEG capture. Do not attribute the whole difference
to the capture kernel, do not claim global AVX-512 acceleration, and do not
enable it by default on this machine. The optional implementation and its tests
remain available for targeted tuning and other CPUs. The default remains AVX2.

The AVX-512-enabled vendor SIMD suite additionally passed 235 tests; the
explicit microbenchmark is the one ignored test. Strict library/CLI Clippy with
the optional feature and formatting checks passed. No CI/platform policy,
encoder behavior, file format, or Python integration changes were made.

## Final Retained Build

The RCT formula and experiment-only maintained tests were removed. The entire
Modular/ANS subtree has no substantive diff from the starting source. Retained
work is JPEG reconstruction improvements, coarse pixel profiling, the optional
AVX-512 feature, checked native AVX2/AVX-512 capture kernels and their tests.
The final default AVX2 executable is `target/release/rs-jxl.exe`, also preserved
as `out/decoder-perf-20261024/rs-jxl-final-avx2.exe`, SHA256
`ca5e5eaeb76baf831c6cc3afecdfec67ed215db32220a408277e65d0eff9c6c5`.

Both final six-JPEG 1T checks passed all 18 original-byte comparisons each.
The corpus is still 43.672832 MPix at 2.4228 bpp; no compressed inputs change.

| Measurement | Before MPix/s | Final AVX2 MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: |
| Unpinned, 5 decodes/batch | 33.820 | 33.364 | 44.125 |
| CPU 4 affinity, 10 decodes/batch | 32.676 | 34.414 | 43.997 |

Reports: `final-avx2/report.json` and `final-avx2-pinned/report.json`. Both use
one excluded warmup and three measured rotating rounds. The latter launches
the benchmark parent with `SetProcessAffinityMask(..., 0x10)`; child decoders
inherit that affinity. No compilation or other benchmark ran concurrently.

These runs disagree on the sign of the aggregate difference (-1.35% versus
+5.32%), and four of six pinned cases were slower even though the largest case
was faster. **Do not claim a stable whole-decoder speedup or universal
non-regression from this kernel change.** The controlled block microbenchmark
supports a local AVX2 CfL improvement, but end-to-end performance remains noisy
and the official single-thread target is not achieved. All per-case regressions
remain in the raw reports; no sample was dropped. The optional full AVX-512
configuration remains disabled by default.

### Further Targets

- Measured: progressive JPEG scan rewriting. The earlier 9oo8k1 profile spent
  130.3 of 205.6 ms in rewriting, more than coefficient capture. Inspect
  `jpeg/coefficient_writer.rs` and `jpeg/entropy_writer.rs` for repeated band
  traversal, zigzag indexing, correction-bit packing and Huffman emission.
  A faster coefficient transpose alone cannot remove this cost.
- Measured only at coarse granularity: ordinary PNG decoding spends about 98%
  of the photo profile inside the vendor backend. Add separate timings for
  entropy reading, predictor reconstruction and inverse transforms before
  choosing the next substantial algorithm change.
- Hypothesis, not yet a measured gain: weighted prediction has four independent
  candidate/error lanes within each pixel. Vectorizing those lanes is more
  plausible than vectorizing adjacent pixels, which depend on previous results.
  Preserve wide signed intermediates and exact rounding.
- Hypothesis: palette reconstruction can repeat index conversion and palette
  access across channels. Non-delta/non-weighted cases must remain distinct
  from stateful delta/weighted cases; skipping prediction updates indiscriminately
  would corrupt later pixels.
- Existing overhead: multithreaded API calls recreate thread pools, while the
  official benchmark reuses its runner. A reusable decoder context could help
  small images without changing arithmetic. Single-thread `Runner` already
  avoids creating a pool, so it is not the explanation for large 1T JPEG costs.

## Stage Hotspot Audit

This phase analyzes hotspots only; no further decoder algorithm or SIMD changes
are made. The new `decoder-profile` feature is disabled by default in both Rust
crates and the maintained backend. It adds no unsafe code or dependencies.

Two attempts to start OS-wide CPU sampling failed: xperf rejected kernel flags
and WPR could not enable the system profiling policy. No policy was modified;
WPR subsequently reported no active recording. Those attempts produced no CPU
samples. The discarded OS-tracing script is not retained. Results below instead
come from direct stage instrumentation, like the earlier C++ analysis.

### Measurement

- Default AVX2 dispatch, one decoder thread, same nine pre-encoded JXL inputs.
- Normal throughput: three uninstrumented batches per case, approximately eight
  seconds each, reporting median wall time divided by decode repetitions.
  Input reading/startup are amortized, no output file is written during timing.
- These completed timing rows are in
  `out/decoder-hotspots-20261024/run1/report.json`. The overall report is marked
  failed because its subsequent OS collector failed, not because decoding or
  verification failed. All nine timing rows completed before that failure.
- Instrumentation: three rounds of five decodes per case, with exact original
  PNG sample / JPEG byte checks before each case. All nine passed.
  Detailed raw output: `out/decoder-hotspots-20261024/stages/report.json`.
- Pixel timing requests float output, matching the earlier djxl no-output
  benchmark; exact integer samples are verified separately. This is not a
  benchmark of the separate native-wide-integer capture API or PNG file writing.
- Instrumented stage times are inclusive and overlapping. They are not added
  together as wall time. Shares below divide matching coarse stage medians by
  the instrumented decode-total median, not by the uninstrumented timing.
- The two builds/timing phases differ in instrumentation and thermal/scheduling
  conditions. Differences between their times are **not optimization gains**.

| Input | JXL bpp | Normal 1T ms | Normal MPix/s | Main instrumented route |
| --- | ---: | ---: | ---: | --- |
| Tecnick RGB_OR_1200x1200_023 | 8.5081 | 279.723 | 5.148 | Weighted flat-tree channel loop: 93.5% |
| news.ycombinator.com | 0.5072 | 153.975 | 12.478 | Weighted flat-tree/WP-only loops: 74.4% |
| actions-mail-send-receive | 1.0387 | 54.818 | 4.782 | Weighted flat-tree channel loop: 91.0% |
| 16-bit vgqcws_vin_709_g1 | 28.8574 | 0.932 | 4.393 | Weighted flat-tree channel loop: 72.4% |
| Paper screenshot 9bde6aaf... | 2.3784 | 297.936 | 4.887 | Weighted flat-tree channel loop: 88.9% |
| wallhaven-49yekn PNG | 2.7327 | 251.565 | 8.243 | Weighted flat-tree/WP-only loops: 84.9% |
| wallhaven-yqxzqx JPEG | 4.3686 | 496.994 | 29.670 | Coefficient backend 60.0%; rewrite 39.1% |
| wallhaven-9oo8k1 JPEG | 0.9593 | 208.048 | 45.361 | Coefficient backend 36.2%; rewrite 63.4% |
| wallhaven-vpyekp JPEG | 0.3020 | 22.108 | 111.162 | Coefficient backend 64.8%; rewrite 32.9% |

The 16-bit image is only 64x64 and is not evidence about large 16-bit throughput.
This is a targeted diagnostic set, not a full-dataset speed comparison.

### What Is Actually Hot

**Pixels:** `decode_modular_channel_impl` selects a concrete decoder type and
times the whole channel loop. The weighted route includes MA property/tree
evaluation, entropy reading, prediction, error updates, and row storage. A 93.5%
route share does **not** mean weighted prediction alone consumes 93.5%.

RCT is only about 0.04..0.20 ms on the larger PNG cases, under 0.1% of decode time.
Palette work is about 4.2 ms for the paper screenshot and 5.5 ms for the web
screenshot, roughly 1.6% and 4.4% respectively. Neither is the leading target in
this set; continuing to tune a small inverse-transform kernel has little upside.

**Progressive JPEG 9oo8k1:** median instrumented total is 199.823 ms. AC refinement
scans consume 83.256 ms (41.7%), initial AC scans 29.322 ms (14.7%), initial DC
7.416 ms and DC refinement 5.996 ms. Rewrite totals 126.779 ms, coefficient
recovery 72.359 ms. This identifies refinement scanning as a substantial target,
not coefficient transpose alone.

**Sequential JPEG yqxzqx:** median instrumented total is 501.788 ms. VarDCT group
processing consumes 219.285 ms (43.7%), and the sequential JPEG scan writer
192.632 ms (38.4%). The VarDCT figure includes coefficient entropy decoding and
capture, not entropy alone. Both sides deserve attention; progressive-only work
will not solve this case.

### Narrow-Probe Limit

`sample.*` timers inspect one call in 1024 per thread/slot, with staggered phases.
They do not time every pixel. However, weighted prediction/error/entropy probes
mostly measure about 30..37 ns against a timer-pair floor of about 27..30 ns.
The inlined arithmetic also has no black-box barriers against compiler code
motion. These values are **not sufficient to rank those tiny functions or split
the channel-loop percentage**. Do not multiply them into a claimed CPU share.
The small image has only about 60 samples per probe and is especially unsuitable.
The broader route/scan results above are the accepted hotspot evidence.

### SIMD and Rewrite Priorities

1. Weighted reconstruction: consider a complete feature-dispatched row kernel,
   not a tiny per-pixel SIMD helper call. Four predictor-error lanes are
   independent within one pixel; four adjacent pixels are not. Prototype
   vectorized error sums/updates and fixed-point arithmetic while retaining i64
   intermediates, u32 wrapping, exact reciprocal lookup, signed rounding, and
   stable property tie handling. Inspect generated assembly before accepting
   microbenchmark claims. Pure Rust source without explicit SIMD can already
   have compiler-generated vector instructions.
2. JPEG AC refinement: classify a block's coefficient magnitudes/bitplanes in
   batches, derive new/existing/zero masks, then preserve ordered sign, ZRL,
   EOB and correction-bit emission. Reducing repeated zigzag walks across scans
   may matter as much as vector width. Bounded memory and byte-identical original
   JPEG reconstruction remain requirements.
3. Sequential JPEG: investigate batched AC classification and symbol emission
   alongside coefficient entropy decoding. Keep this separate from progressive
   tuning; their measured bottlenecks differ.
4. Secondary candidates: already-decoded reference-channel properties and
   non-overlapping LZ77 history chunks can be SIMD-friendly. Their isolated
   shares are not established by this run, so they are not yet measured leaders.

Single-stream ANS has a serial state/renormalization dependency; dynamic MA
contexts depend on reconstructed pixels. Merely replacing the loop with wider
loads cannot remove those dependencies. No approximate float arithmetic or
out-of-bounds loads are justified by these performance targets.

### Reproduction and Checks

```powershell
$env:CARGO_TARGET_DIR = 'out/decoder-hotspots-20261024/target'
cargo build --release --locked --offline -p rs-jxl-cli --features decoder-profile
$env:RS_JXL_PROFILE = '1'
& "$env:CARGO_TARGET_DIR/release/rs-jxl.exe" decode <input.jxl> --num_threads=1 --disable_output --bits=32 --num_reps=5 --quiet
# For JPEG-byte reconstruction use --reconstruct_jpeg instead of --bits=32.
Remove-Item Env:RS_JXL_PROFILE
```

`tools/profile_decoder_stages.py` reuses the completed local timing rows, verifies
inputs/decoded output, and stores three diagnostic rounds without repeating the
normal throughput benchmark. It requires the detailed build feature and rejects
missing backend diagnostics. Its two parser tests pass; strict Clippy for the
library/CLI with `--all-targets --features decoder-profile -- -D warnings` passes.
The nine exact roundtrips also pass. No full regression, platform acceptance,
or decoder performance improvement is claimed by this instrumentation-only work.

## Refinement Masks and Exact Weight Lookup

### JPEG Refinement SIMD

`jpeg/refinement.rs` classifies the 64 integer coefficients into zigzag-order
nonzero, newly-nonzero, correction-bit and positive-sign masks. Runtime AVX2
dispatch processes sixteen coefficients at a time; the portable implementation
produces the same masks. Loads stay inside a local initialized `[i16; 64]`;
there are no overlapping 32-bit gathers beyond the last i16. Precision checking
rejects out-of-range magnitudes, including `i16::MIN`, only inside the selected
spectral band. DC semantics are unchanged.

The writer visits only set nonzero bits, preserving the original ZRL, sign,
correction and EOB order. BMI2 PEXT packs suffix corrections where available,
with a scalar fallback. Failed speculative entropy batches still replay the
original scalar emission path, preserving partial-error behavior and poisoning.

The isolated candidate is `out/decoder-perf-20261024/rs-jxl-refinement-avx2.exe`,
SHA256 `c4fd36de07de5cad0f1c7629fea6fdf45de7af7e4208df7c64851fa4daad82d6`.
The comparison is against `rs-jxl-final-avx2.exe` (`ca5e5eae...`), not an
instrumented binary. `refinement-simd/report.json` records six real JPEGs, one
and eight requested threads, one excluded warmup, three rotating rounds and ten
in-process reconstructions per command. All 36 original-JPEG byte checks pass.

The progressive `wallhaven-9oo8k1.jpg` is 9.437184 MPix, with a fixed input JXL
of 0.9593 bpp. These are reconstruction timings, not IDCT rendering or encoding:

| Threads | Before ms | After ms | Official ms | Before MPix/s | After MPix/s | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 205.350 | 154.239 | 160.474 | 45.957 | 61.186 | 1.331x |
| 8 | 77.271 | 52.531 | 114.697 | 122.130 | 179.649 | 1.471x |

This is one progressive performance sample, not evidence of a broad JPEG gain.
The five sequential controls have 1T speedups of 0.989..1.006x and 8T speedups
of 0.959..0.996x. Do not describe the latter as proven non-regression. Their
entropy algorithms were not modified; final combined-build checks follow below.
All JXL inputs are reused unchanged, so bpp cannot change in this decoder test.

Focused Release JPEG tests: 328 passed, nine explicitly ignored external-oracle
tests, zero failures. Coverage includes scalar/SIMD masks for every spectral
subband and bit plane, every i16 value's precision classification, pending EOB
table changes, output limits, poisoning, correction packing, signed coefficients,
restart boundaries and parallel scan reconstruction. The whole test binary's
ThinLTO build took about ten minutes; tests themselves took 0.15 seconds. This
compilation cost is not included in any decoder performance number.

### Common Pixel-Lossless Weight Lookup

The weighted channel route combines prediction, MA contexts, entropy parsing,
error updates and stores; its large profile share is not a measurement of the
weight calculation alone. The new private `weighted_weights.rs` helper replaces
four repeated error normalization/reciprocal/multiplication sequences with exact
table loads when all four errors are below 256 and header weights are below 16.
The read-only 16 KiB table is generated at compile time; no runtime allocation,
float approximation, unsafe indexing or additional per-pixel CPU dispatch is
introduced. Full-range errors/weights use the original integer formula.

The final default-AVX2 candidate is
`out/decoder-perf-20261024/rs-jxl-weighted-lookup.exe`, SHA256
`63e1e882edbd331514790c21a794faf9af77a4fe0ebdfa6827fdec572d7ccf76`.
The normal `target/release/rs-jxl.exe` has the same contents. Combined results
are in `weighted-lookup/report.json` and `weighted-lookup/REPORT.md`: ten PNGs
and six JPEGs, one/eight threads, three rotating measured rounds, five
in-process decodes per command and one excluded warmup. All 96 original-output
checks pass. PNG checks compare integer samples including invisible RGB; JPEG
checks compare original file bytes. All bpp values use the unchanged JXL input.

PNG timing requests float32 buffers to match the existing djxl no-output
benchmark, not native-integer capture or PNG file writing. Process startup and
input IO are amortized, not removed. In particular, the 64x64 image's ~9 ms here
is dominated by startup with only five decodes; it must not be compared to the
earlier ~0.9 ms calibrated long-loop result as a decoder regression.

| PNG workload | Threads | Before total ms | After total ms | Official total ms | Throughput ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ten images, sum of per-image medians | 1 | 1261.186 | 1238.061 | 1343.759 | 1.019x |
| Ten images, sum of per-image medians | 8 | 317.562 | 312.639 | 258.359 | 1.016x |

Selected 1T details from that same run (not isolated predictor timings):

| Input | bpp | Before ms | After ms | Before MPix/s | After MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| kodim19.png | 8.9082 | 76.958 | 73.629 | 5.109 | 5.340 |
| RGB_OR_1200x1200_023.png | 8.5081 | 238.452 | 233.402 | 6.039 | 6.170 |
| news.ycombinator.com.png | 0.5072 | 122.335 | 120.192 | 15.705 | 15.985 |
| actions-mail-send-receive.png | 1.0387 | 56.210 | 53.330 | 4.664 | 4.916 |
| 9bde6aaf...png | 2.3784 | 263.952 | 259.874 | 5.517 | 5.603 |
| wallhaven-49yekn.png | 2.7327 | 234.008 | 230.325 | 8.861 | 9.003 |

This is a small observed gain, not a claim that weighted prediction or MA
traversal has been solved. Eight-thread pixel decoding still trails official
djxl in this aggregate; several sequential JPEGs also remain slower. The
combined progressive JPEG result is 206.388 -> 158.161 ms (1.305x) at 1T and
83.334 -> 56.708 ms (1.470x) at 8T. Do not attribute that JPEG gain to the lookup;
the separate refinement-only run above establishes the substantial change.

Checks for the combined source:

- Six bounded lookup-oracle tests: all small errors/all u8 weights, mixed lanes,
  fallback in each lane, exponent boundaries, wrapped sums and seeded full-range
  errors. The harness imports the actual helper and supplies an independently
  generated reciprocal table; production-parent integration is tested below.
- 29 decoder API tests, seven wide-integer tests, 17 streaming tests and four
  hostile-input parent tests pass. The hostile subprocess entry point is
  intentionally marked ignored in ordinary discovery and exercised by parents.
- Strict parent-workspace Clippy with `--all-targets --features decoder-profile
  -- -D warnings` passes. These benchmarks use default builds without profiling.
- Running the vendored jxl crate's own unit-test target was blocked by 37 missing
  `include_bytes!` test resources (for example `basic.jxl` and fuzz fixtures).
  No files were fabricated or upstream tests disabled. The self-contained
  parent tests above ran instead; this is not full upstream conformance testing.

Saved comparison checkpoints remain in the ignored `out` directory. No commit/push,
decoder protocol change, encoding decision, CI change, ARM kernel, WSL or
external profiling service was involved.

### Longer Control Recheck

The initial short combined run showed small slowdowns on three 8T controls.
`weighted-controls/report.json` repeats only those cases, serially, with five
rotating measured rounds, twenty in-process decodes per command and one excluded
warmup. All nine additional original-output checks pass. The slowdowns did not
reproduce; this supports startup/scheduling variation rather than a demonstrated
persistent regression, but is not a guarantee for arbitrary images or CPUs.

| Input | Threads | Before ms | After ms | After MPix/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| kodim19.png | 8 | 17.709 | 17.299 | 22.730 | 1.024x |
| news.ycombinator.com.png | 8 | 32.293 | 30.841 | 62.296 | 1.047x |
| wallhaven-3q3z9d.jpg | 8 | 51.989 | 51.715 | 121.404 | 1.005x |

Both bounded changes are retained. Main-workspace formatting and `git diff
--check` pass. No further progressive JPEG optimization was started; the next
performance focus remains common pixel-lossless and sequential JPEG decoding.
