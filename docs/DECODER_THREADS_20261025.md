# Multithreaded Pixel Decode, 2026-10-25

## Retained Change

Completed frames now own the typed buffers into which the decoder writes.
Previously, backend `OwnedRawImage` buffers were copied row by row into a second
set of U8/U16/F32 vectors after each frame. `decode/buffers.rs` now allocates
zeroed typed planes, exposes borrowed byte views to the backend, and transfers
those planes into the completed frame. This removes a serial, whole-frame copy
from both batch and streaming decoding, without adding a second thread pool.

Native-endian U16/F32 storage is requested explicitly. Allocation and size
checks remain fallible. Partial snapshots still copy: callers can retain a
snapshot while subsequent input changes the active frame. Completed frames
also remain independent of later frames. Exact wide-integer capture is not
changed. There is no encoder, compressed-stream or public API change.

The safe typed allocation/cast helpers use bytemuck 1.25.2, which was already
present transitively in the lockfile. It is now an explicit dependency of
`rs-jxl` with its allocation feature enabled.

This removes one duplicate output allocation, not the decoder's working
images or reference frames. For the tested 4096x4096 RGBA8 case, the main plane
and separately returned alpha total 80 MiB; one duplicate of those buffers is
eliminated. This is an allocation-accounting result, not a measured total-RSS
reduction.

## Measurement

Default Release build, AVX2-capable, Ryzen 9 7945HX, Windows. Builds and timed
experiments were serialized. All streams were reused, not re-encoded for each
candidate. Original PNG samples, including invisible RGB, were checked for
each binary and thread count before timing.

`tools/benchmark_pixel_threads.py` measures native U8/U16 output using the
CLI's internal decode timer. It excludes process startup and input-file IO;
it includes decoder-context construction, output allocation and decoding.
Three rotating before/after rounds follow one warmup round. Large images use
five in-process repetitions per round; controls use ten. Values are medians,
not confidence intervals. Timed decodes do not write images.

Official `djxl --disable_output` forces F32, so it is not a valid direct
comparison to these integer-output times. Any official comparison below is
separately labelled F32. Historical wall-time-per-repetition benchmarks also
have a different measurement boundary.

File hashes protect test-input and executable identity only. Correctness is
decoded samples or reconstructed original JPEG bytes, never matching newly
encoded JXL bytes. bpp here describes an unchanged input stream.

## Large Integer Outputs

The two large streams were created once by official libjxl 0.12.0 using
`-d 0 -e 5 --num_threads=8 --keep_invisible=1`. This selects decoder workloads;
it is not an encoder compression comparison.

| Image | MPix | bpp | Threads | Before ms | After ms | Before MPix/s | After MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wallhaven-mlzoy1.png | 8.2944 | 13.742297 | 1 | 843.067 | 854.546 | 9.838 | 9.706 |
| wallhaven-mlzoy1.png | 8.2944 | 13.742297 | 8 | 135.138 | 140.114 | 61.377 | 59.197 |
| wallhaven-mlzoy1.png | 8.2944 | 13.742297 | 16 | 90.369 | 82.089 | 91.783 | 101.041 |
| 146845733_p0.png | 16.7772 | 0.005563 | 1 | 1024.657 | 999.293 | 16.373 | 16.789 |
| 146845733_p0.png | 16.7772 | 0.005563 | 8 | 179.229 | 155.140 | 93.608 | 108.142 |
| 146845733_p0.png | 16.7772 | 0.005563 | 16 | 122.637 | 101.015 | 136.804 | 166.086 |

The RGBA case improves throughput by 15.5% at eight threads and 21.4% at
sixteen. The natural image improves by 10.1% at sixteen but is 3.6% slower at
eight and 1.3% slower at one. Do not describe this as a universal speedup.

## Integer Controls

| Image | Threads | bpp | Before ms | After ms | Before MPix/s | After MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB_OR_1200x1200_023.png | 1 | 8.5081 | 225.067 | 223.736 | 6.398 | 6.436 |
| RGB_OR_1200x1200_023.png | 8 | 8.5081 | 37.147 | 36.421 | 38.765 | 39.538 |
| RGB_OR_1200x1200_023.png | 16 | 8.5081 | 27.880 | 25.310 | 51.650 | 56.894 |
| Blobby_volley_2_alpha_6.png | 1 | 3.5546 | 30.528 | 30.871 | 15.638 | 15.464 |
| Blobby_volley_2_alpha_6.png | 8 | 3.5546 | 7.371 | 7.140 | 64.765 | 66.858 |
| Blobby_volley_2_alpha_6.png | 16 | 3.5546 | 7.578 | 7.069 | 62.999 | 67.535 |
| news.ycombinator.com.png | 1 | 0.5072 | 102.017 | 96.554 | 18.833 | 19.898 |
| news.ycombinator.com.png | 8 | 0.5072 | 20.948 | 17.072 | 91.714 | 112.535 |
| news.ycombinator.com.png | 16 | 0.5072 | 16.827 | 14.381 | 114.178 | 133.599 |
| clover.png | 1 | 12.7831 | 183.367 | 183.944 | 5.718 | 5.701 |
| clover.png | 8 | 12.7831 | 30.220 | 31.182 | 34.699 | 33.628 |
| clover.png | 16 | 12.7831 | 19.644 | 18.825 | 53.378 | 55.702 |
| grayscale.png (cvo9xd) | 1 | 2.8849 | 12.260 | 12.012 | 20.391 | 20.812 |
| grayscale.png (cvo9xd) | 8 | 2.8849 | 4.058 | 3.813 | 61.601 | 65.563 |
| grayscale.png (cvo9xd) | 16 | 2.8849 | 4.066 | 3.989 | 61.490 | 62.675 |
| vgqcws_vin_709_g1.png | 1 | 28.8574 | 0.852 | 0.864 | 4.806 | 4.742 |
| vgqcws_vin_709_g1.png | 8 | 28.8574 | 0.957 | 0.997 | 4.280 | 4.108 |
| vgqcws_vin_709_g1.png | 16 | 28.8574 | 0.969 | 1.005 | 4.225 | 4.078 |
| paper screenshot (9bde6aaf...) | 1 | 2.3784 | 254.182 | 248.921 | 5.729 | 5.850 |
| paper screenshot (9bde6aaf...) | 8 | 2.3784 | 41.291 | 40.394 | 35.265 | 36.048 |
| paper screenshot (9bde6aaf...) | 16 | 2.3784 | 33.070 | 31.120 | 44.032 | 46.791 |
| wallhaven-49yekn.png | 1 | 2.7327 | 215.135 | 209.635 | 9.639 | 9.891 |
| wallhaven-49yekn.png | 8 | 2.7327 | 36.696 | 32.974 | 56.507 | 62.887 |
| wallhaven-49yekn.png | 16 | 2.7327 | 27.048 | 24.671 | 76.662 | 84.051 |

All controls are eight-bit except the 64x64 sixteen-bit `vgqcws` image. That
small image regresses by about 0.01-0.04 ms and does not benefit from extra
threads. `clover` also regresses at eight threads (about 3.1% throughput).

Aggregate throughput is total pixels divided by the sum of image-median
decode times, not the average of per-image MPix/s. Combining the separately
run large-image and control sets gives the following ten-image subset:

| Set | Threads | Before MPix/s | After MPix/s | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| Eight controls | 1 | 8.473 | 8.615 | +1.68% |
| Eight controls | 8 | 48.526 | 51.008 | +5.12% |
| Eight controls | 16 | 63.254 | 68.617 | +8.48% |
| All ten | 1 | 11.671 | 11.797 | +1.08% |
| All ten | 8 | 68.436 | 72.526 | +5.98% |
| All ten | 16 | 96.383 | 109.033 | +13.12% |

The largest improvement is not from parallelizing a single entropy stream.
Removing the serial output copy matters more as group decoding becomes faster
with additional workers. Prediction and entropy dependencies remain.

## Matched F32 Official Comparison

The separate `compare_decoder_checkpoints.py` run uses F32 for all three
decoders, ten in-process decodes, three rotating rounds and one warmup. These
times are process wall time divided by repetitions: startup and input IO are
amortized, not excluded. Original PNG integer samples were checked separately.

| Image | Threads | Before ms | After ms | Official 0.12 ms | After MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB_OR_1200x1200_023.png | 1 | 235.718 | 235.511 | 265.746 | 6.114 | 5.419 |
| RGB_OR_1200x1200_023.png | 8 | 45.480 | 42.053 | 43.020 | 34.242 | 33.472 |
| RGB_OR_1200x1200_023.png | 16 | 34.737 | 30.873 | 32.840 | 46.643 | 43.849 |
| news.ycombinator.com.png | 1 | 122.946 | 113.920 | 143.108 | 16.865 | 13.425 |
| news.ycombinator.com.png | 8 | 36.791 | 26.814 | 31.043 | 71.650 | 61.890 |
| news.ycombinator.com.png | 16 | 33.307 | 23.752 | 24.753 | 80.889 | 77.617 |
| paper screenshot (9bde6aaf...) | 1 | 268.264 | 264.688 | 303.931 | 5.501 | 4.791 |
| paper screenshot (9bde6aaf...) | 8 | 54.010 | 46.584 | 52.796 | 31.258 | 27.580 |
| paper screenshot (9bde6aaf...) | 16 | 46.884 | 42.738 | 45.271 | 34.071 | 32.165 |

Rust is faster in this three-image comparison, but this is not evidence of
matching or exceeding official performance over all JXL inputs.

A JPEG-transcoded pixel control (`wallhaven-vpyekp.jpg`, 2.4576 MPix,
0.301989 bpp) at one thread measured 37.037 -> 29.521 ms, or
66.356 -> 83.248 MPix/s; official measured 28.103 ms / 87.450 MPix/s.
This uses the same wall-time/F32 convention. Rust U8 and F32 samples are exactly
unchanged, and original JPEG bytes reconstruct exactly. Existing differences
from official rendering remain: at most one U8 level and about 1.43e-6 in F32
on this image with the default `jxl-rs` dither. The harness's
`completed_with_pixel_differences` status records those official differences;
the before/after exactness fields were independently required to pass.

## Correctness Coverage

Focused Rust targets on the retained source passed **98 tests, 2 ignored**:
`decode` (31), `decode_dither` (4), `decode_jpeg` (8), `decode_streaming` (17),
`decode_wide_integer` (7), `decoder_context` (12), `decoder_parallel_pixels` (5),
`decoder_weight_lookup` (7), `decode_hostile` (4), and buffer unit tests (3).
The ignored entries are an externally supplied JPEG corpus test and the
hostile-test child entry point (invoked by its parent tests).

The new integration coverage checks actual multigroup decoding at 1/2/8
threads, odd-width edge groups, RCT, weighted prediction, palette, squeeze,
U8/U16, hidden RGB, mixed extra channels, streaming and retained multiframe
outputs. Original samples are the oracle, not merely single-thread output.
Buffer unit tests also check pointer ownership transfer, independent partial
snapshots, native endian storage, limits, signed zero and float NaN payloads.
The timing parser has two Python unit tests for valid and invalid timer lines.

Real-image verification passed for all ten PNG originals at 1/8/16 threads
on both executables. These tests do not claim full upstream conformance or
replace a broad malformed-input fuzzing campaign.

Workspace/all-target Clippy passed with `decoder-avx512,decoder-profile` and
warnings denied. Scoped rustfmt checks and `git diff --check` passed. The
default Release binary matches the frozen direct-output executable below.
Optional AVX512 was compile-checked here, not separately performance-tested.
No commit or push was made.

## Rejected Candidates

Two earlier candidates passed correctness checks but did not establish a
stable benefit on seven PNG controls at 1/8/16 threads:

| Candidate | 1T throughput change | 8T change | 16T change |
| --- | ---: | ---: | ---: |
| Large-stream-first group order, concurrency floor, parallel output copy | +0.35% | +0.96% | -1.23% |
| 1024-entry weighted lookup and parallel output copy | +0.21% | -1.28% | +0.61% |

Both runtime experiments were removed. The spatial task order and previous
concurrency logic remain. The weighted lookup remains 256 entries (16 KiB),
not 1024 entries (64 KiB). Expanded exact-formula regression tests remain.
These two early reports used wall time divided by repetitions; their absolute
times must not be mixed with the internal-timer tables above.

## Artifacts and Reproduction

Reports are under `out/parallel-pixels-20261025/`:

- `direct-large/report.json`: large integer outputs and exact sample checks.
- `direct-controls/report.json`: integer controls.
- `direct-f32/report.json`: matched F32 official comparison.
- `direct-jpeg-pixels/report.json`: JPEG reconstruction and U8/F32 control.
- `cost-copy-native/report.json`: rejected task-order/copy candidate.
- `lookup-copy-native/report.json`: rejected lookup/copy candidate.

Frozen executables:

- Before: `out/entropy-20261025/rs-jxl-restored.exe`, SHA256
  `597e1bfb8f30c1c4ef974b9604c91ee94de2f47f7d8abd791bebb1e242e7a4f1`.
- After: `out/parallel-pixels-20261025/rs-jxl-direct-output.exe`, SHA256
  `c4d8ae0577e8680c8d18c5dddec253a46dfe27862c6eafb5547522653940e44f`.

Run from this repository with an existing output parent and a new output leaf:

```powershell
python -B tools/benchmark_pixel_threads.py `
  --before out/entropy-20261025/rs-jxl-restored.exe `
  --after out/parallel-pixels-20261025/rs-jxl-direct-output.exe `
  --threads 1 8 16 --repeats 3 --warmups 1 --decode-reps 5 `
  --out out/parallel-pixels-20261025/recheck `
  --case out/parallel-pixels-20261025/natural4k.jxl ../jxl_python/test_img/wallhaven-mlzoy1.png `
  --case out/parallel-pixels-20261025/rgba4k.jxl ../jxl_python/test_img/mt_lay/t1/146845733_p0.png
```

This is a development-host subset. It does not establish cross-platform
runtime parity, full-corpus performance, or a general benefit from increasing
thread counts beyond the available groups and memory bandwidth.
