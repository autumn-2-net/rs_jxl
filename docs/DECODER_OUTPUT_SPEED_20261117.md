# Decoder Output and JPEG Reconstruction, 2026-11-17

The baseline is `fdbcc42edd588f92dbec9e94bd40cb5e822ae0fe`, the previous
Modular reader optimization, committed and pushed to `origin/main` before
this round. This round covers pixel-lossless Native output, exact original-JPEG
reconstruction, JPEG-transcoded VarDCT rendering, floating-point lossless input
and multiframe output. It changes neither encoding decisions nor JXL syntax.

## Retained Changes

- `decode/native.rs`: after the existing geometry and byte-budget checks, move
  a wide grayscale color plane directly into `PixelData::U32` when no alpha is
  present. A narrow main plane with a wide extra channel keeps its conversion.
  When alpha and color depths match, use the original alpha sample directly;
  differing depths retain the original integer scaling formula. Extra-channel
  ordering, authoritative planes and the output toggle are preserved.
- Vendored `render/low_memory_pipeline/save/identity.rs`: fuse native-endian
  RGB/RGBA U16 conversion and interleaved storage using the existing quantizer
  and SIMD primitives. Other formats/layouts retain their fallback. This does
  not change rounding, add dithering or introduce unsafe code.

## Rejected JPEG Candidate

Serial scan emission was changed to append directly to the final output buffer,
avoiding a separate scan Vec and its copy. Both an extra-field writer (V1) and a
same-size writer (V2) were tested. V2 preserved scan-relative public output-limit
errors and passed 410 selected JPEG unit tests. Correct output alone did not
establish a speed improvement: in both V2 eight-thread rounds, the large JPEG
case slowed down (0.961x and 0.930x), while the complete JPEG group fell from
133.285 to 127.419 MPix/s. One-thread aggregate improved from 33.576 to 34.377.

The JPEG candidate was removed from the working tree. Final V3 builds retain
only the pixel-output changes above; JPEG source matches the baseline. Earlier
patches, binaries and measurements remain in the local evidence folder. The
observed regressions do not establish a cause such as layout/register effects.

## Fixed Inputs and Measurement

Windows x86_64, Ryzen 9 7945HX, rustc 1.97.1, release thin LTO with one codegen
unit. Before and after CLI/FFI builds use the identical Cargo command and
features. The public Python `Decoder` uses each actual CFFI library with its
own reusable context/thread pool. Calls are sequential and alternate library
order; no codec calls are mocked. Timers include returned arrays/bytes and
FFI overhead, excluding context construction, IO and validation.

`tools/benchmark_decoder_outputs.py` reads a fixed manifest. It does no input
discovery or encoding. Every warmup and measured result is checked outside the
timer, and library/input SHA256 identities are recorded and rechecked. Expected
lossless arrays and JPEG bytes come from the original sources, not from an
after-build decode. VarDCT render expectations explicitly use the before
decoder: equality there checks unchanged rendering, not JPEG-source pixel
losslessness or a new independent conformance result.

The 29 development cases comprise:

- Six existing official libjxl e9 pixel streams: Kodak, webpage, transparent
  icon, gray photo, small RGB16 photo and PDF/text; 4,286,834 spatial pixels.
- RGB16/RGBA16 tiles, lossless F32, Gray31/RGBA31 tiles, and three RGBA16 frames.
  Transparent RGB values are retained. New fixture encoding uses only the
  frozen before DLL, outside all decoder timers.
- Eleven JPEG pairs: three existing 2.46/9.44/14.75 MP natural-image streams,
  baseline/progressive 420 and grayscale, baseline 444/422/440, and restart
  markers. Every reconstructed byte must match the original JPEG.
- Six render cases: U16 and F32 output from the small natural JPEG stream,
  progressive 420 and baseline 422.

These are focused development samples, not a previously unused holdout corpus.
The same images intentionally appear under different operations/output types.
Each build/case/thread setting has two warmups and nine measured calls per
round. Two rounds are retained. One-thread runs use affinity mask `4`; eight
threads use mask `0x5555`. The displayed time is the arithmetic mean of the
two per-round medians, giving both rounds equal weight despite substantial
latency changes between them. Raw trials and each round's median remain in
JSON. Throughput is spatial pixels / elapsed seconds; multiframe cases count
all returned full frames. Aggregate rates divide total pixels by summed times,
not by an average of individual MPix/s figures.

## Final Results

Final V3 results are mixed; this is not a general decoder speedup acceptance.
Two-round aggregate throughput:

| Operation | Threads | Before MPix/s | After MPix/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Six real-image pixel cases (subset) | 1 | 12.270 | 12.096 | -1.4% |
| All 11 lossless pixel cases | 1 | 13.511 | 13.442 | -0.5% |
| 11 original-JPEG reconstructions | 1 | 42.878 | 42.309 | -1.3% |
| Six U16/F32 render cases | 1 | 82.667 | 83.543 | +1.1% |
| Three RGBA16 frames | 1 | 36.751 | 37.275 | +1.4% |
| Six real-image pixel cases (subset) | 8 | 55.350 | 56.037 | +1.2% |
| All 11 lossless pixel cases | 8 | 51.378 | 53.223 | +3.6% |
| 11 original-JPEG reconstructions | 8 | 155.702 | 150.815 | -3.1% |
| Six U16/F32 render cases | 8 | 131.865 | 129.054 | -2.1% |
| Three RGBA16 frames | 8 | 62.227 | 58.471 | -6.0% |

The real-image group is a subset of the 11 pixel cases, not an additional
corpus. Gray31 shows the clearest repeatable benefit: **27.557 -> 29.696
MPix/s (1T, +7.8%)** and **48.517 -> 59.137 (8T, +21.9%)**. Its individual
rounds improved 7.2..8.4% at 1T and 12.8..32.6% at 8T, so the magnitude is
variable. RGB16 tiles improved 3.1% at 8T but were unchanged at 1T. The large
U16 render case improved only 1.2% / 0.8% in the final two-round summary;
that does not establish a broadly useful U16 output speedup.

Regressions remain visible: 8T multiframe throughput fell 6.0%, the combined
render group fell 2.1%, and original-JPEG reconstruction fell 3.1% even after
the JPEG source was restored. One-thread aggregate lossless pixels fell 0.5%.
These measurements must not be reported as universal gains or assumed to be
noise. They require further isolation before accepting the whole candidate as
a general performance improvement. Earlier binaries/runs cannot be substituted
for the final source merely because their results look better.

Per-case measurements below include every final sample, including controls,
small cases, unchanged paths and regressions. A ratio above 1 means faster.

### 1 threads

| Case | Before ms | After ms | Before MPix/s | After MPix/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| real-photo-kodak | 63.297 | 63.533 | 6.212 | 6.189 | 0.996x |
| real-web | 60.100 | 60.103 | 31.968 | 31.966 | 1.000x |
| real-icon-alpha | 35.826 | 37.558 | 7.317 | 6.980 | 0.954x |
| real-gray | 11.925 | 11.709 | 20.965 | 21.352 | 1.018x |
| real-photo16-small | 0.988 | 0.996 | 4.145 | 4.112 | 0.992x |
| real-pdf-regression | 177.246 | 180.512 | 8.215 | 8.067 | 0.982x |
| rgb16-tiles | 68.162 | 68.175 | 15.384 | 15.381 | 1.000x |
| rgba16-hidden-tiles | 86.517 | 87.240 | 12.120 | 12.019 | 0.992x |
| rgb-f32-lossless | 74.132 | 74.685 | 14.145 | 14.040 | 0.993x |
| gray31-tiles | 38.051 | 35.310 | 27.557 | 29.696 | 1.078x |
| rgba31-hidden-tiles | 30.855 | 30.626 | 8.496 | 8.559 | 1.007x |
| rgba16-three-frames | 16.049 | 15.824 | 36.751 | 37.275 | 1.014x |
| jpeg-small | 15.605 | 15.758 | 157.484 | 155.958 | 0.990x |
| jpeg-small-render-u16 | 24.245 | 23.946 | 101.366 | 102.631 | 1.012x |
| jpeg-small-render-f32 | 34.040 | 33.714 | 72.197 | 72.895 | 1.010x |
| jpeg-medium | 139.818 | 139.623 | 67.496 | 67.590 | 1.001x |
| jpeg-large | 465.879 | 474.273 | 31.651 | 31.091 | 0.982x |
| jpeg-baseline-420 | 0.333 | 0.335 | 46.355 | 46.161 | 0.996x |
| jpeg-progressive-420 | 0.431 | 0.434 | 35.822 | 35.579 | 0.993x |
| jpeg-progressive-420-render-u16 | 0.477 | 0.475 | 32.390 | 32.516 | 1.004x |
| jpeg-progressive-420-render-f32 | 0.474 | 0.475 | 32.605 | 32.540 | 0.998x |
| jpeg-baseline-444 | 0.422 | 0.422 | 36.607 | 36.607 | 1.000x |
| jpeg-baseline-422 | 0.341 | 0.353 | 45.295 | 43.817 | 0.967x |
| jpeg-baseline-422-render-u16 | 0.491 | 0.482 | 31.479 | 32.084 | 1.019x |
| jpeg-baseline-422-render-f32 | 0.478 | 0.483 | 32.299 | 32.011 | 0.991x |
| jpeg-baseline-440 | 0.352 | 0.353 | 43.892 | 43.817 | 0.998x |
| jpeg-baseline-gray | 0.274 | 0.282 | 56.284 | 54.836 | 0.974x |
| jpeg-progressive-gray | 0.362 | 0.366 | 42.733 | 42.202 | 0.988x |
| jpeg-restart-dri20 | 0.379 | 0.386 | 40.792 | 40.052 | 0.982x |

### 8 threads

| Case | Before ms | After ms | Before MPix/s | After MPix/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| real-photo-kodak | 15.997 | 15.805 | 24.581 | 24.879 | 1.012x |
| real-web | 13.219 | 13.624 | 145.345 | 141.021 | 0.970x |
| real-icon-alpha | 10.762 | 10.638 | 24.359 | 24.642 | 1.012x |
| real-gray | 3.751 | 3.716 | 66.642 | 67.280 | 1.010x |
| real-photo16-small | 1.051 | 1.056 | 3.896 | 3.878 | 0.995x |
| real-pdf-regression | 32.670 | 31.661 | 44.570 | 45.992 | 1.032x |
| rgb16-tiles | 15.023 | 14.566 | 69.797 | 71.988 | 1.031x |
| rgba16-hidden-tiles | 20.411 | 20.200 | 51.374 | 51.911 | 1.010x |
| rgb-f32-lossless | 20.293 | 20.417 | 51.672 | 51.357 | 0.994x |
| gray31-tiles | 21.613 | 17.731 | 48.517 | 59.137 | 1.219x |
| rgba31-hidden-tiles | 15.385 | 14.863 | 17.039 | 17.637 | 1.035x |
| rgba16-three-frames | 9.479 | 10.087 | 62.227 | 58.471 | 0.940x |
| jpeg-small | 8.189 | 8.069 | 300.103 | 304.554 | 1.015x |
| jpeg-small-render-u16 | 12.977 | 12.868 | 189.384 | 190.981 | 1.008x |
| jpeg-small-render-f32 | 22.750 | 23.676 | 108.027 | 103.800 | 0.961x |
| jpeg-medium | 48.361 | 48.629 | 195.140 | 194.066 | 0.994x |
| jpeg-large | 112.222 | 117.620 | 131.397 | 125.366 | 0.954x |
| jpeg-baseline-420 | 0.351 | 0.355 | 44.061 | 43.509 | 0.987x |
| jpeg-progressive-420 | 0.446 | 0.447 | 34.602 | 34.541 | 0.998x |
| jpeg-progressive-420-render-u16 | 0.499 | 0.496 | 30.934 | 31.162 | 1.007x |
| jpeg-progressive-420-render-f32 | 0.518 | 0.522 | 29.812 | 29.584 | 0.992x |
| jpeg-baseline-444 | 0.450 | 0.445 | 34.368 | 34.696 | 1.010x |
| jpeg-baseline-422 | 0.438 | 0.445 | 35.282 | 34.692 | 0.983x |
| jpeg-baseline-422-render-u16 | 0.499 | 0.500 | 30.974 | 30.881 | 0.997x |
| jpeg-baseline-422-render-f32 | 0.500 | 0.502 | 30.897 | 30.746 | 0.995x |
| jpeg-baseline-440 | 0.363 | 0.363 | 42.515 | 42.597 | 1.002x |
| jpeg-baseline-gray | 0.291 | 0.299 | 53.065 | 51.664 | 0.974x |
| jpeg-progressive-gray | 0.371 | 0.374 | 41.622 | 41.338 | 0.993x |
| jpeg-restart-dri20 | 0.410 | 0.415 | 37.706 | 37.233 | 0.987x |


## Verification

- 54 related integration tests passed across wide integer, parallel pixel,
  streaming, layer/preview/orientation, lossless edge, JPEG and dither targets.
- `check_u16_output` verified 283,140 boundary words plus 283,140 original
  normalized integer words per Scalar/SSE4.2/AVX implementation. Fused output
  equals the existing planar quantizer for each implementation. Original
  integer words are also equal across implementations. Pre-existing scalar
  half-way rounding differences are documented separately from this change.

- Final-source public JPEG checks were rerun: 8 `decode_jpeg` and 1
  `decode_jpeg_capture` tests passed (already included in the 54 above). Two
  explicit external-input tests remained ignored.
- Rust 1.89.0 `cargo check` passed for the library, the two changed integration
  targets and `check_u16_output`. Rustfmt, Python syntax and `git diff --check`
  passed for the changed files.
- All 2,552 final before/after benchmark calls passed exact output checks.
  All 29 cases ran at 1/8 threads in both rounds; input and DLL hashes stayed
  unchanged. This count includes warmups and timed calls, not extra unit tests.

## Evidence and Reproduction

Local artifacts are in `out/decoder-paths-20261117/`. `before.dll`/`before.exe`
are the frozen baseline; `after-v3.dll`/`after-v3.exe` are the final candidate.
`manifest.json` identifies all inputs and oracle types; `fixtures/` contains
the frozen JXL, original expected arrays and JPEG bytes. These artifacts are
not committed. `baseline-build.json`, benchmark reports, `summary-v3.json`,
test logs and the scalar/SIMD check log provide the local receipts. Initial
`after.dll`/`pinned-*.json` and `after-v2.dll`/`v2-*.json` files describe the
rejected JPEG candidates and must not be confused with the final variant.

```powershell
cargo build --offline --release -p rs-jxl-cli -p rs-jxl-ffi
python tools/benchmark_decoder_outputs.py --before out/decoder-paths-20261117/before.dll --after out/decoder-paths-20261117/after-v3.dll --manifest out/decoder-paths-20261117/manifest.json --threads 1 --repeats 9 --warmups 2 --affinity 4 --out out/decoder-paths-20261117/new-1t.json
python tools/benchmark_decoder_outputs.py --before out/decoder-paths-20261117/before.dll --after out/decoder-paths-20261117/after-v3.dll --manifest out/decoder-paths-20261117/manifest.json --threads 8 --repeats 9 --warmups 2 --affinity 0x5555 --out out/decoder-paths-20261117/new-8t.json
cargo run --offline -p rs-jxl --example check_u16_output
```

Repeat each benchmark with a fresh report filename for the second round.
Byte-reversible JPEG reconstruction is measured separately from pixel rendering.
No lossy-encoder, other-platform or whole-corpus performance claim is implied.
