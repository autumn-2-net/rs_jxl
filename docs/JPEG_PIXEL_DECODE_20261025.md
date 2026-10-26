# JPEG-Transcoded JXL Pixel Decoding

## Separate Contracts

Earlier JPEG performance reports measured reconstruction of the original JPEG
file, not conversion of that JXL into pixels. This report measures the latter.
Byte-reversible JPEG transcode preserves original coefficients and reconstruction
data. It does not force independent renderers to use identical IDCT, upsampling,
color arithmetic or output quantization.

Original integer lossless JXL still requires exact original samples, including
invisible RGB. Original-JPEG reconstruction still requires identical JPEG bytes.
Those acceptance rules are not weakened by reporting VarDCT rendering differences.

## Why The Tables Differed

The pinned official Rust source (`5ccf8694e3779e2931ca487d46251e3050bf849b`)
already contained the different table. Before this compatibility change,
our imported `util/dither.rs` and `dither_32x32.bin` were unchanged.

- [PR #841](https://github.com/libjxl/jxl-rs/pull/841) originally ported the C++
  dithering and explicitly noted small VarDCT U8 discrepancies even with that
  table, attributed to different floating-point values.
- [Commit 8188a9f](https://github.com/libjxl/jxl-rs/commit/8188a9f07b4d508f37f6b4c6cb5d2b80b6226bec)
  moved the table to compact U8 storage to reduce binary size.
- [PR #940](https://github.com/libjxl/jxl-rs/pull/940) intentionally replaced the
  pattern with DZgas perfect-best-blue-noise for more even dithering. The merge
  message notes a perfectly uniform value distribution.

Thus the different pattern is an upstream rendering choice, not evidence of
damaged JPEG coefficients. Our local C++-compatible table is a compatibility
choice, not a claim that the Rust upstream pattern has worse visual quality.
The selectable table is only used for floating-point-to-U8 output; native integer
output, F32 output and original-JPEG reconstruction bypass it. Modular
smooth-squeeze retains its independent upstream table.

## Selecting The Output Pattern

Default behavior remains `jxl-rs`. Both modes are per-decoder options, not a
global variable or environment toggle. Save stages select an immutable table
once, without branching on the mode for every pixel. Previews, orientations,
extra output planes and both backend render pipelines propagate the selection.
Explicit CMS/background postprocessing still uses its existing nondithered final
quantizer; this option selects the backend's already-dithered output only.

```powershell
.\target\release\rs-jxl.exe decode input.jxl output.png --dither jxl-rs
.\target\release\rs-jxl.exe decode input.jxl output.png --dither libjxl
```

```rust
use rs_jxl::decode::{DecodeOptions, DitherMode, SampleFormat, decode};

let options = DecodeOptions {
    format: SampleFormat::U8,
    dither: DitherMode::Libjxl,
    ..DecodeOptions::default()
};
let image = decode(&input_bytes, &options)?;
```

This is a quantizer-pattern choice, not a promise of bit-exact VarDCT rendering
against libjxl. Small floating-point pipeline differences remain.

## Before Compatibility Change

Six frozen JXL inputs, five sequential JPEG sources and one progressive source
(`wallhaven-9oo8k1.jpg`). One thread, one excluded warmup, three alternating
measured rounds, ten full decodes per process. Timing is wall time divided by
repetitions, including amortized startup/input IO and allocation. It excludes
PNG encoding or output file writes. Aggregate throughput is total megapixels
divided by the sum of per-image median times.

| Output | Rust seconds | Official 0.12 seconds | Rust MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: | ---: |
| F32 pixels | 0.924465 | 0.787580 | 47.241 | 55.452 |

Rust took 17.38% longer in this bounded corpus. Individual U8 pixel comparisons
differed in 49.00061..70.34952% of pixels (any RGB component), with maximum
component error 1. F32 maximum absolute errors were 9.54e-7..3.00e-6.
All six original JPEG files reconstructed byte-for-byte in both decoders.

These are pixel-rendering measurements, not the earlier JPEG byte-reconstruction
measurements and not a result for all possible JPEGs.

## Controlled Quantizer Replay

For `wallhaven-yq5e9x.jpg` (2,459,648 pixels), independently render both U8 and
F32 output, then reapply output quantization outside the codecs:

| Comparison | Differing pixels | Maximum U8 difference |
| --- | ---: | ---: |
| Original native U8 outputs | 64.43698% | 1 |
| Both F32 outputs, same C++ table | 0.0020328% (50 pixels) | 1 |
| Both F32 outputs, no dither | 0.0009351% (23 pixels) | 1 |
| Rust F32 + upstream Rust table vs Rust U8 | 0 | 0 |
| Official F32 + C++ table vs official U8 | 0 | 0 |

The exact same-backend replays establish that different tables account for the
dominant difference on this image. The remaining 50 pixels cross quantization
thresholds because their underlying floats differ. No tolerances are used to
relabel these samples as exactly equal.

## Repeated Scanning

For these commands the CLI directly calls `decode::Decoder::decode` once per
requested benchmark repetition. It does not call `reconstruct_jpeg`, decode a
restored JPEG, or run a second complete `inspect`. Output is disabled, so the
selected-frame metadata sidecar inspection is not taken either. Backend section
processing consumes completed LF/HF sections and tracks completed passes.

The repeated sequential-JPEG AC validation/nonzero scan discussed in the
reconstruction optimization is therefore not executed by this pixel path.
This source audit is not a measured hotspot breakdown of VarDCT pixel rendering;
it does not prove there is no other redundant work. `--num_reps=10` is deliberate
benchmark repetition, not accidental decoding ten times per API call.

## Sequential JPEG Reconstruction Optimization

The sequential writer merges AC range validation and nonzero-position extraction.
An AVX2 kernel produces a checked zigzag mask; unsupported CPUs use a scalar
implementation. Emission visits nonzero coefficients and emits ZRL gaps/EOB.
DC validation, output budgeting and missing-Huffman-code errors retain their
existing checks/fallback. Progressive writing is unchanged.

Six original JPEG files reconstruct byte-for-byte with both Rust checkpoints
and official djxl. Serial rotating timing, one thread, three measured rounds,
five decodes per process, one excluded warmup. Final build, after discarding
the separate pixel candidates:

| Source | bpp | Before ms | After ms | Official ms | After MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| wallhaven-yqxzqx.jpg | 4.3686 | 521.447 | 446.655 | 404.618 | 33.013 |
| wallhaven-9oo8k1.jpg (progressive) | 0.9593 | 160.025 | 157.779 | 161.944 | 59.813 |
| wallhaven-vpyekp.jpg | 0.3020 | 27.108 | 24.033 | 18.485 | 102.260 |
| wallhaven-yq5e9x.jpg | 1.7473 | 56.239 | 47.193 | 36.436 | 52.119 |
| wallhaven-jexkwm.jpg | 1.4847 | 175.857 | 149.160 | 129.811 | 55.607 |
| wallhaven-3q3z9d.jpg | 2.3864 | 182.621 | 155.855 | 140.317 | 40.284 |

Sequential cases use 11.3..16.1% less total reconstruction time; the progressive
control is effectively unchanged. These timings are JPEG bytes, not pixels.
Aggregate: before 1.123298 s / 38.879 MPix/s, final 0.980675 s / 44.533 MPix/s,
official 0.891611 s / 48.982 MPix/s. Final Rust throughput improved 14.5% over
the previous Rust checkpoint, but is not yet equal to official on this corpus.
The first ten-decodes/run experiment independently showed a 12.4% throughput
improvement; raw results of both runs are retained.

## Pixel Stage Profile

Feature-gated instrumentation on three images, five F32 decodes each, one thread:

| Scope (ms/decode) | yqxzqx | 9oo8k1 | jexkwm |
| --- | ---: | ---: | ---: |
| Complete pixel decode | 396.768 | 157.756 | 172.028 |
| Output copy | 24.890 | 17.669 | 15.437 |
| VarDCT group | 241.320 | 64.932 | 80.347 |
| Block entropy (inside group) | 183.873 | 35.834 | 45.227 |
| Dequantization + IDCT (inside group) | 40.376 | 17.831 | 25.153 |
| Post-IDCT pipeline | 55.973 | 45.435 | 34.200 |

Timers overlap by scope and add measurement overhead, especially per block;
do not sum them or substitute these numbers for uninstrumented performance.
F32 output of the diagnostic binary was verified against the accepted Rust
checkpoint before profiling. Entropy work remains a larger cost than IDCT.

## Pixel Candidates Not Retained

Three candidates were tested together: typed output-row copies, removal of one
nested 8x8 IDCT dispatch, and balanced selection of the Modular weighted-error
context property. The earlier accepted weight lookup optimization is separate
and remains intact.

Old/new/official ran serially in rotating order with identical F32 output
settings. On six JPEG-transcoded inputs, the old checkpoint totaled 0.934527 s,
the candidate 0.927436 s, and official 0.789449 s. The apparent 0.76% candidate
improvement is too small to establish a useful win; `jexkwm` was 3.28% slower.
Default-mode U8 and F32 decoded samples were identical to the older Rust
checkpoint on all six inputs. This compares decoded pixels, not JXL file bytes.

Four original-PNG controls also passed exact integer comparison. Candidate
speedups were 1.024x (kodim19), 1.005x (web screenshot), 1.003x (clover), and
0.998x (PDF screenshot). Aggregate changes were noise-scale. All three candidate
code changes were removed; the row-copy boundary regression test was retained.
These candidate timings are not advertised as improvements in the final build.

The selectable `libjxl` pattern was independently checked on the same six real
JPEG-transcoded inputs: U8 differing-pixel percentages versus official were
0.00252%, 0.00432%, 0.00203%, 0.00203%, 0.00776%, and 0.00514%, respectively.
Maximum channel difference remained 1. F32 rendering errors were unchanged.

## Reproduction And Artifacts

`tools/benchmark_jpeg_pixel_decode.py` times F32 only (`--bits 32`, the default)
and independently verifies JPEG bytes, U8 pixels and F32 pixels using actual
temporary output files. libjxl's `djxl_main.cc` supplies only `JXL_TYPE_FLOAT`
formats when `filename_out` is empty; `--disable_output --bits_per_sample=8`
therefore does NOT benchmark U8 output. The tool rejects this invalid comparison.
`--dither jxl-rs|libjxl` selects the Rust output pattern; omit it for older
checkpoints which do not implement the option. Official commands are unchanged.
`--before OLD_RUST_EXE` adds a third, rotating checkpoint and compares decoded
U8/F32 samples with it. The older checkpoint receives no new dither option.
Its `--case-label` filter selects a subset without re-encoding source inputs.
`tools/analyze_jpeg_pixel_quantization.py` performs the bounded quantizer replay.

The benchmark records actual commands, binary identities, trials, bpp, MPix/s
and all differences. Temporary validation PAM/PFM/JPEG files are removed.
Original inputs are unchanged; no new image encoding is required.

- `out/jpeg-pixels-20261025/report.json`: original F32 timings and U8/F32 checks.
- `out/jpeg-pixels-20261025/REPORT.md`: per-image initial results.
- `out/jpeg-pixels-20261025/quantizer-before/report.json`: controlled replay.
- `out/jpeg-pixels-20261025/sequential-mask/report.json`: checked-mask comparison.
- `out/jpeg-pixels-20261025/sequential-final/report.json`: final retained-code check.
- `out/jpeg-pixels-20261025/stages-before/report.json`: diagnostic stage timings.
- `out/jpeg-pixels-20261025/pixels-after/report.json`: libjxl-table real-image checks.
- `out/jpeg-pixels-20261025/pixels-paired/report.json`: rejected pixel candidates.
- `out/jpeg-pixels-20261025/png-controls/report.json`: rejected-candidate controls.
- `out/jpeg-pixels-20261025/final-u8/report.json`: final U8/F32/JPEG validation;
  explicitly marked INVALID for Rust-versus-official timing because djxl used
  F32, not U8. Do not cite it as a speedup. Rust-old/current U8 timings alone
  were 31.907/32.687 ms and 121.412/119.912 ms (noise-scale changes).

The final retained source passed 71 decoder integration tests and strict
workspace/all-target Clippy including `decoder-profile`. During this change,
330 JPEG unit tests, 76 decoder-related unit tests, two focused CLI option tests
and 24 benchmark/diagnostic Python tests also passed. The final Release was
built offline and checked with the six real JPEG inputs above. External-fixture tests are ignored
by ordinary discovery; real-image verification is recorded separately above.
This is not a complete upstream conformance run. Vendored jxl tests with omitted
external fixtures are not built by ordinary root-workspace testing.
