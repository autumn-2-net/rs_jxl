# Parallel JPEG Reconstruction: 2026-10-24

## Scope

Second decoder performance iteration, following
[the initial writer optimization](DECODER_PERFORMANCE_20261024.md).
The target is official 0.12 throughput on identical JXL streams with eight
requested threads. Single-thread results are measured separately, not hidden.
This is original-JPEG byte reconstruction, not IDCT/pixel rendering or PNG
decoding. The encoder, compressed size and lossy policy are unchanged.

## Implementation

- Reuse the decoding call's existing Rayon pool for reconstruction. No second
  pool, global worker cache, new dependency, unsafe code or C++ runtime.
- Sequential scans without restarts are divided on MCU boundaries. Every
  fragment starts with the preceding MCU's exact per-component DC predictor.
  Complete fragments retain tail-bit counts; the merge removes local stuffing,
  joins the actual bit sequences, then applies final stuffing and original
  padding. It does not byte-concatenate arbitrarily aligned entropy streams.
  Stuffing-free runs merge word-wise; shifted bytes still receive fresh stuffing.
  Partition multiplication uses a widened intermediate, including on 32-bit hosts.
- Sequential scans with restart markers split only at restart boundaries.
  DC predictors start at zero; original RST indexes and extra marker fill are
  preserved. Byte-aligned fragments can then be directly concatenated.
- Progressive scans encode independently from the complete coefficient arrays.
  A single work queue avoids barriers between long refinement scans; outputs
  and source marker records are assembled in their original order.
- The restart/progressive parallel paths require implicit all-one padding.
  Explicit/mixed padding keeps the existing serial path. Small scans and
  single-worker calls avoid parallel work. Nonrestart sequential fragments can
  preserve explicit padding because alignment happens once after merging.
- Temporary fragment byte limits sum to at most one output budget. A skewed or
  tight per-fragment budget falls back to the bounded serial writer. Final
  output and aggregate padding limits remain enforced. This bounds scratch
  payloads, not total process RSS or allocator capacity. A failed speculative
  attempt can repeat scan work once; it cannot retry indefinitely.
- A bounded 512-byte scratch writer uses a 64-bit reservoir and commits once
  per block/correction batch, emitting up to eight bytes per reservoir flush.
  Failed batches replay scalar emission, preserving
  per-symbol failure state, DC history and error poisoning.
- JPEG coefficient capture computes quantization ratios once per frame and
  caches chroma scales per group/factor. Zero factors bypass prediction, not
  coefficient range checks. Group geometry is cached, and group-buffer
  allocation no longer holds the shared capture-state mutex. Signed division,
  both fixed-point roundings and intermediate overflow checks are retained.
- JPEG coefficient reading dispatches ANS/Huffman/LZ77 outside the symbol loop,
  reuses nonzero contexts, and uses checked i32 addition for unshifted passes.
  Shifted passes retain checked wide arithmetic. Completion validates every DC
  but avoids a second strided write pass when its correction offset is zero.

Normal use needs no new flag: `DecodeOptions.threads` / CLI `--threads` controls
the existing pool. The public coefficient-only reconstruction helper retains
its serial behavior; pool plumbing is internal to the decoder.

## Verification

- 318 core JPEG-focused tests passed, with nine existing optional external
  encoder-oracle tests ignored. New checks exercise every pending-bit alignment,
  byte stuffing, continuous per-component DC history, subsampled MCU layouts,
  restart fill/order, nonzero restart predictor resets, short final intervals,
  overflow-safe partition arithmetic and tight output/padding limits. A skewed
  progressive scan explicitly verifies successful serial fallback when one
  worker's budget is insufficient but the complete output still fits.
- 14 maintained vendor JPEG-capture tests passed, including differential CfL
  arithmetic for all 256 factors, extreme quantization/coefficient values,
  cache invalidation, transposition and error precedence.
- Four vendor group tests passed, including differential context, coefficient,
  partial-state and error comparisons for the specialized JPEG coefficient loop.
- 44 decoder integration tests passed across ordinary decode, JPEG reconstruction,
  DC refinement and hostile input. The hostile worker is ignored as a standalone
  test and invoked by its parent tests with process isolation; the optional
  externally supplied JPEG corpus test was not run.
- Direct vendor `cargo test` cannot compile without the excluded upstream image
  fixtures. The vendor tests above used the existing ignored fixture-enabled
  `out/decoder-vendor-validation` copy after synchronizing the current maintained
  source. No fixtures were added to the maintained vendor or Git.

Original JPEG bytes, not new/old JXL byte identity, are the archival correctness
criterion. No timeouts or incorrect cases may be silently excluded from results.

Focused checks (from the independent repository root):

```text
cargo test --locked --offline -p rs-jxl --lib jpeg::
cargo test --locked --offline -p rs-jxl --test decode --test decode_jpeg --test jpeg_dc_refinement --test decode_hostile
cargo clippy --locked --offline -p rs-jxl -p rs-jxl-cli --all-targets -- -D warnings
cargo fmt --all -- --check
python -B -m unittest discover -s tools -p test_compare_decoder_checkpoints.py
```

The fixture-copy vendor checks used `--no-default-features --features
sse42,avx,brotli`, separately filtering `frame::jpeg::tests` and
`frame::group::test`. Full upstream corpus tests, Linux and macOS runtime tests
were not performed; this change neither starts WSL nor changes CI.

## Measurement Method

Native Windows x86_64 on Ryzen 9 7945HX; Rust 1.97.1, normal Cargo Release,
thin LTO and one codegen unit. Fixed checkpoint before this iteration:
`out/decoder-perf-20261024/rs-jxl-optimized.exe` (SHA256
`0a4590c4f1bed48a9665dd5bb56c765b705c2bac558e125fbe8307ab4eea72ec`).

The same official-0.12 e9 JPEG-transcode streams go to each decoder. Processes
run serially in rotating order, with profiling disabled, one excluded warmup,
three measured invocations and ten in-process decodes per invocation. Reported
time is median process wall/repetitions, including amortized startup/input IO,
allocation and pool setup, but excluding output-file writes and verification.
Official retains a runner across repetitions; Rust still creates one per API
call, although it now reuses that pool across coefficient and writer stages.

The three original diagnostic samples are measured at one and eight threads.
Three additional JPEGs were selected from sorted top-level `test_img/*.jpg`,
excluding the original three, using Python `random.Random(20261024).sample(...,3)`:
`wallhaven-yq5e9x`, `wallhaven-jexkwm`, and `wallhaven-3q3z9d`. They were selected
without consulting performance and are measured at both thread counts. They are a
small held-out check, not an unbiased estimate of all JPEG archives.

Aggregate MPix/s is total pixels divided by the sum of per-image median seconds,
not the mean of per-image rates. bpp is `8 * JXL bytes / original pixels`.

Driver: [compare_decoder_checkpoints.py](../tools/compare_decoder_checkpoints.py).
Supply repeated `--case JXL ORIGINAL` pairs and a new `--out` directory; it
verifies original bytes for every decoder/thread/input pair before timing.

Final measured binary SHA256:
`4f4f4276a25db97f3961d824e3364af533dce36706a8dcd867a4794a27a30ff8`.
Official djxl SHA256:
`c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.
The driver checks binary/input identities again at completion; these identities
pin the experiment and are not a requirement that compressed output stay fixed.

## Results

All six cases, one and eight requested threads: 36 original-JPEG byte checks
and 144 timed/warmup process batches passed. No failures or timeouts were skipped.
This corpus totals 43.672832 MPix and 13,226,053 compressed bytes: **2.4228 bpp**.

Eight-thread results, ten decodes per process:

| Input | bpp | Before ms | After ms | Official ms | After MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wallhaven-yqxzqx | 4.3686 | 337.649 | 124.823 | 173.745 | 118.132 | 84.869 |
| wallhaven-9oo8k1 | 0.9593 | 195.771 | 75.288 | 113.765 | 125.347 | 82.953 |
| wallhaven-vpyekp | 0.3020 | 21.591 | 15.112 | 12.095 | 162.625 | 203.190 |
| wallhaven-yq5e9x | 1.7473 | 40.895 | 21.777 | 20.891 | 112.945 | 117.736 |
| wallhaven-jexkwm | 1.4847 | 124.413 | 65.634 | 66.988 | 126.374 | 123.819 |
| wallhaven-3q3z9d | 2.3864 | 121.261 | 59.714 | 67.062 | 105.142 | 93.620 |

Aggregate throughput over the same six images:

| Threads | Before MPix/s | After MPix/s | Official MPix/s | After vs before | After vs official |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 22.797 | 33.339 | 50.018 | 1.462x | 0.667x |
| 8 | 51.894 | 120.527 | 96.080 | 2.323x | 1.254x |

Eight-thread aggregate time falls **56.9%** versus the previous Rust checkpoint,
and is **20.3% lower** than official 0.12. Four of six eight-thread medians beat
official. However, vpyekp still takes 24.9% longer and yq5e9x 4.2% longer; the
2.1% advantage on jexkwm is small and should not be treated as a robust margin.
Single-thread time is still 50.0% above official in aggregate. Therefore this
is **aggregate eight-thread parity exceeded, not universal decoder parity**.
No claim is made for PNG pixel decoding, unsupported JPEG modes, other CPUs,
or whole-dataset throughput from these six diagnostic cases.

Raw medians/ranges, commands, identities and every trial:
`out/decoder-perf-20261024/final-parallel-comparison/report.json` and `REPORT.md`.
Retained executable: `out/decoder-perf-20261024/rs-jxl-parallel-final.exe`.

Single-thread stage diagnostics, separate ten-decode runs with profiling enabled:

| Input | Total ms/decode | Coefficient stage ms | Writer ms |
| --- | ---: | ---: | ---: |
| wallhaven-jexkwm | 227.704 | 163.375 | 61.490 |
| wallhaven-9oo8k1 | 219.083 | 74.759 | 142.532 |

The remaining bottleneck differs by JPEG structure: coefficient capture/decoding
dominates the sequential held-out case, while progressive scan writing dominates
9oo8k1. More thread dispatch alone cannot fix the single-thread deficit. These
inclusive stage values are diagnostic, not replacements for the comparison above.

### Short-Run Amortization Check

The two short-running eight-thread cases were also run with **100 decodes per
process**, keeping one excluded warmup, three rotating-order measurements and
the same binaries/inputs. This reduces startup/IO amortization effects; it does
not remove per-call pool construction or allocation. Six byte checks and 24
timed/warmup process batches passed.

| Input | Before ms | After ms | Official ms | After MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| wallhaven-vpyekp | 16.366 | 10.263 | 10.535 | 239.457 | 233.291 |
| wallhaven-yq5e9x | 36.994 | 19.204 | 20.224 | 128.078 | 121.617 |

These small advantages (2.6% and 5.3% throughput) indicate batch parity on these
cases, not a universal margin. They also show why short process-wall batches
must not be called isolated kernel measurements. Launch amortization and steady
execution conditions affect the result; no fixed launch cost was subtracted.
Do not mix these two rows into the ten-repetition aggregate or discard the
slower short-batch results. The single-thread deficit remains unchanged.

Raw evidence: `out/decoder-perf-20261024/small-amortization/report.json`.
