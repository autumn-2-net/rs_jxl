# Pixel Lossless Codec Speed, 2026-11-09

## Scope And Retained Changes

This round targets pixel-lossless encoding and native integer pixel decoding,
not byte-reversible JPEG reconstruction. The frozen pre-change executables are
based on `352b129`, with the earlier JPEG work present on both sides. The same
Windows Release configuration is used: thin LTO, one codegen unit, default
runtime SIMD features, Ryzen 9 7945HX. No new CPU instruction requirement,
effort reduction, sampling cap, search restriction or public API is introduced.

- Encoder `modular/weighted.rs`: store each pixel's four unsigned predictor
  errors together. Four history allocations become one; collecting three
  neighbors and updating four lanes no longer traverses four separate vectors.
  Prediction, signed error history, u32 wrapping, tie order and reset semantics
  are unchanged. The independent full-history i128 reference remains the oracle.
- Decoder `frame/modular/predict.rs`: when all five neighbors are equal and all
  four signed errors are zero, all sub-predictors and the final clamped prediction
  equal that neighbor. Skip weight calculation in this case, but retain every
  normal error update, including when the decoded pixel breaks the flat run.
- `tools/benchmark_pixel_calls.py`: paired reusable CFFI contexts with original
  integer arrays as the oracle. Record whole-call timing, native encoding time,
  bytes and bpp separately. Decode timing always uses the same frozen stream.

The decoder shortcut is an exact algebraic simplification, not an approximation
or an assumption that pixels near one another must match. Compressed JXL byte
identity is not an optimization target or a correctness gate.

## Measurement

All pixel encoding uses e9, `source_search=False`, eight threads and integer
NumPy input. This is the current API's ordinary source route, including its
internal tool selection; it is not the optional full outer candidate scheduler.
The four primary images have six measured repetitions per build; seven holdout
images have four. Each has one warmup per build and alternating execution order.

API timings exclude input IO, context construction, hashing and verification.
Encoding includes array transport/output copies; the JSON additionally records
`report.codec_seconds`. Decoding includes the returned NumPy array. Both native
libraries cross-decode every encoded candidate outside the encoder timer, and
every timed decode is checked against the original samples, including hidden RGB.

### In-Memory API Results

Times are medians. Each row's bpp and encoded byte count were unchanged by this
round. MPix counts spatial pixels, not channels.

| Sample | bpp | Encode seconds, before -> after | After encode MPix/s | Decode ms, before -> after | After decode MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB_OR_1200x1200_023.png | 8.5050 | 1.1984 -> 1.1437 | 1.259 | 43.381 -> 43.636 | 33.00 |
| news.ycombinator.com.png | 0.4617 | 0.5652 -> 0.5764 | 3.333 | 25.354 -> 19.416 | 98.95 |
| 9bde6aaf390109f97622d35ff1e900ee.png | 2.2665 | 1.9634 -> 1.9434 | 0.749 | 66.700 -> 51.501 | 28.27 |
| wallhaven-mlzoy1.png | 13.6754 | 6.7372 -> 6.7089 | 1.236 | 256.205 -> 250.270 | 33.14 |
| kodim19.png | 8.8897 | 1.0331 -> 1.0616 | 0.370 | 18.110 -> 18.406 | 21.36 |
| Blobby_volley_2_alpha_6.png | 3.5529 | 0.2452 -> 0.2441 | 1.956 | 8.847 -> 8.509 | 56.10 |
| clover.png | 12.7677 | 1.2904 -> 1.2755 | 0.822 | 37.472 -> 37.446 | 28.00 |
| actions-mail-send-receive.png | 1.0395 | 0.8004 -> 0.7395 | 0.354 | 19.086 -> 17.138 | 15.30 |
| cvo9xd_keong_macan_grayscale.png | 2.8872 | 0.2442 -> 0.2495 | 1.002 | 5.790 -> 5.986 | 41.76 |
| vgqcws_vin_709_g1.png | 28.8613 | 0.0422 -> 0.0416 | 0.099 | 1.177 -> 1.291 | 3.17 |
| wallhaven-49yekn.png | 2.7293 | 1.2340 -> 1.1687 | 1.774 | 44.530 -> 38.737 | 53.53 |

The sum of per-image medians decreases 15.354 -> 15.153 s for encoding (1.013x)
and 0.527 -> 0.492 s for decoding (1.070x). These are weighted by actual time,
not arithmetic averages of per-image speedup percentages.

The encoding result is too small relative to timing variation to establish a
general speedup. For example, primary Tecnick encoding samples span 1.068-1.273 s
before and 0.970-1.322 s after. The history layout is simpler and needs fewer
allocations, but MA/LZ77 still dominate complete encoding. The clear gain is
flat-heavy decoding: about 1.30x on the webpage and PDF in this API matrix.
Natural photos largely stay near their previous speed; tiny inputs are dominated
by fixed overhead. Do not interpret the 64x64 example as sustained throughput.

### Official 0.12 Streams

Ten existing official e9 streams were decoded by both CLI checkpoints at one
and eight threads, with four measured rounds, one warmup and three internal
decodes per round. This timer excludes startup/IO but includes decoder-context
construction, unlike the reusable API above. Original PNG pixels were checked.

Selected single-thread results:

| Sample | Before ms | After ms | After MPix/s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Tecnick | 238.509 | 233.496 | 6.167 | 1.021x |
| Webpage | 105.728 | 71.542 | 26.855 | 1.478x |
| PDF | 268.889 | 209.914 | 6.937 | 1.281x |
| Alpha icon | 48.499 | 40.895 | 6.410 | 1.186x |
| Kodak | 66.465 | 67.136 | 5.857 | 0.990x |

There are real tradeoffs, not a blanket no-regression claim. An eight-thread
reusable-context recheck of the same official streams, with two warmups and ten
alternating measured calls, gave Kodak 17.893 -> 18.412 ms (2.9% more time) and
game UI 8.538 -> 8.876 ms (4.0% more time). Gray improved 4.620 -> 4.503 ms.
The volatile wallhaven-49 CLI result did not reproduce with reusable contexts:
43.905 -> 40.533 ms. Its original CLI ranges were 79-167 ms before and 47-198 ms
after; those medians cannot support a reliable regression conclusion.

## Correctness And Rejected Experiments

- 47 targeted native tests passed: weighted state/history (23), source costs
  (8), lossless edges (3), wide integer decode (7), parallel pixels (5), and the
  new weighted decoder matrix (1 test containing 180 combinations).
- The new matrix covers 8/16/31-bit RGBA, invisible RGB, narrow row boundaries,
  all five weighted presets, WP-only/mixed trees, and ANS/prefix coding.
- 16 benchmark-harness tests passed, including metadata nesting, actual call
  order, cross-decode routing, verification outside timers and failure reports.
- The primary/holdout API reports contain 400 successful original-sample checks.
- Official libjxl 0.12 and 0.13 decoded all 11 new e9 outputs: 22 exact original
  integer-pixel checks passed. This independently checks the new encoded streams.
- Release CLI/FFI build and focused Clippy with `-D warnings` passed.

Small-node MA predictor-pair fusion and decoder second-row-load elision were
rejected after no convincing whole-call benefit. A WP context lookup table was
also removed after unfavorable/noisy whole-encode results. They are not enabled
as hidden switches or retained as default-path experiments.

Local detail instrumentation still points to MA histogram collection/split
scoring on natural images and LZ77 optimal parsing on the PDF. Their inclusive
parallel scope totals overlap and must not be summed into wall time. This round
does not claim to eliminate those dominant encoding costs.

## Evidence And Reproduction

Local reports under `out/pixel-speed-20261109/`:

- `pixel-api-final-r2.json`, `pixel-api-holdout.json`: API samples, sizes and checks.
- `official-stream-decode/report.json`: fixed official streams and 1T/8T timings.
- `official-reusable-recheck.json`: the targeted slower/noisy-case recheck.
- `official-roundtrip.json`: independent 0.12/0.13 validation of new outputs.
- `encoder-profile-before.json`, `encoder-detail-before.json`: diagnostic scopes.

The earlier `pixel-api-r0.json` and `pixel-api-flat-r1.json` are rejected/interim
experiments, not the final result. Other projects were observed compiling during
part of the earlier work; no unrelated processes were terminated. Even the final
paired samples have variation, retained in the JSON rather than hidden.

Use Python with NumPy, CFFI and the benchmark PNG reader dependencies installed:

```powershell
python -B tools/benchmark_pixel_calls.py --before out/pixel-speed-20261109/before.dll --after target/release/rs_jxl_ffi.dll --case ../jxl_python/test_img/wallhaven-mlzoy1.png --effort 9 --threads 8 --repeats 6 --warmups 1 --out out/pixel-speed-recheck.json
python -B -m unittest discover -s tools -p test_benchmark_pixel_calls.py
cargo test --offline -p rs-jxl --lib modular::weighted
cargo test --offline -p rs-jxl --lib modular::cost::tests
cargo test --offline -p rs-jxl --test decoder_weighted_flat --test decode_wide_integer --test decoder_parallel_pixels --test decode_lossless_edges
```

This is targeted Windows validation, not a full-corpus or cross-platform run.
No new lossy-quality, JPEG reconstruction or metadata benchmark is claimed here.
