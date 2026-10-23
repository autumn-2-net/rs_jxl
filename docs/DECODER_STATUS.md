# Maintained Rust Decoder

## Acceptance Target

The first decoder milestone targets the practical capabilities of official
`djxl`: decode still images and visible animation frames, write integer/float
pixels, and reconstruct original JPEG bytes when reconstruction data is present.
It is not satisfied by parsing headers or by returning a re-encoded JPEG.
Full CLI flag parity and full format coverage are separate checks, not assumptions
made from adopting an upstream decoder.

Runtime decoding uses the maintained official `libjxl/jxl-rs` sources in
`vendor/jxl-rs`, pinned to `5ccf8694e3779e2931ca487d46251e3050bf849b`.
The four supporting crates are local path dependencies, not an opaque registry
codec. Upstream BSD notices, PATENTS and AUTHORS are retained. See the
[provenance and patch ledger](../vendor/jxl-rs/PROVENANCE.md) and the
[pre-integration audit](DECODER_BACKEND_AUDIT.md).

No C++ linkage or runtime codec subprocess is introduced. The encoder remains
the direct migration of our optimized C++ encoder. Independent official `djxl`
executables are development test oracles only.

## Pixel API

```rust
use rs_jxl::decode::{decode, decode_frame, DecodeOptions, SampleFormat};

let options = DecodeOptions { threads: 4, ..Default::default() };
let image = decode(&jxl_bytes, &options)?;
for frame in &image.frames {
    // frame.pixels: typed interleaved Gray/GA/RGB/RGBA samples.
    // frame.extra_planes: all extra channels in header order, including alpha.
    // frame.duration_ticks: integer ticks in image.info.animation's timebase.
}
let selected = decode_frame(&jxl_bytes, 2, &options)?;
```

`decode` consumes a complete input slice, validates all frames and collects
requested trailing metadata. `decode_frame` reconstructs dependencies before
the selected zero-based *visible* frame and stops there. It does not validate
the rest of the file, does not promise arbitrary O(1) seeking, and sets
`complete = false`. Tail metadata is deliberately not reported for an early
return. Invisible zero-duration layers are composited internally; they do not
each become a returned frame. This matches ordinary rendered-frame decoding,
not a raw-layer extraction API.

`SampleFormat::Native` preserves source integer ranges through 16 bits. An
integer declared as 10 bits is returned in U16 storage with values 0..1023.
Floating input is widened to F32. Integers wider than 16 bits are normalized
F32, which **cannot preserve every U31 integer**. `U8` and `U16` explicitly
request conversion to their full storage ranges; `F32` requests float output.
The decoder never labels such format conversion pixel-bit-exact.

Profile declarations are checked by default. Some legacy official fixtures use
Main10 complexity without a `jxll` marker, accepted by djxl but refused by strict
upstream Rust checks. `allow_implicit_main10 = true` (CLI
`--allow-implicit-level10`) enables that compatibility explicitly; input,
sample, output and metadata budgets remain in force.

All extra planes are rendered at canvas resolution. Their source depth, name,
type, `dim_shift`, alpha association, spot color and CFA index are exposed in
`image.info.extra_channels`. Interleaved alpha uses the main image's output
depth; for differently sized alpha use its separate native plane as the
authoritative samples. `include_extra_channels = false` suppresses planar
copies, not the alpha component of RGBA/GA output.

Orientation is applied by default. Set `adjust_orientation = false` to keep
codestream order. Alpha association is kept as encoded; invisible RGB is not
cleared. High-precision rendering is enabled to match libjxl's default
`JXL_HIGH_PRECISION=1`, unlike upstream Rust's default low-precision spline
truncation. Spot-color rendering defaults on as in djxl; disable
`render_spot_colors` for the unmodified base color planes. Embedded and output
color profiles are distinct fields and must not be confused after color
conversion. CMYK is not represented by the project's RGB-only PixelLayout;
the Black plane remains available as an extra channel.

## Resource Scope

Defaults: 512 MiB input, 512 MiB retained output samples, 64 MiB expanded
metadata, 64 million pixels, 256 million backend samples and 10,000 visible
frames. Thread count is 1..256 with an owned pool; one thread runs inline.
Arithmetic and allocations are checked. Compressed auxiliary boxes are
expanded with an explicit byte ceiling.

These are not a total process-memory or time sandbox. The backend also retains
reference frames, entropy buffers, ICC data and parser state. Aligned working
outputs coexist with returned samples. Raw metadata can be buffered by the
upstream parser before the outer expansion limit is checked. Limits must be
set conservatively for untrusted data; a process boundary is still advisable
for hostile inputs.

## CLI and JPEG Reconstruction

```text
rs-jxl decode input.jxl output.png --threads 4
rs-jxl decode animation.jxl frames.npy --bits 32
rs-jxl decode animation.jxl selected.png --frame 2
rs-jxl decode archive.jxl original.jpg
rs-jxl decode - - --format npy --bits 32
```

PNG writes visible animation frames as APNG when the timebase is exactly
representable; it does not silently discard other frames. NPY stores main color
channels followed by each extra channel once, including alpha. PPM/PGM/PFM
require one compatible frame; they cannot silently discard alpha/extra planes.
The CLI rejects unsupported sample/metadata representations rather than
implicitly quantizing float data. File output is published transactionally;
existing files require `--force` and cannot alias the input.

`decode::reconstruct_jpeg(&bytes, &DecodeOptions::default())` returns the
original JPEG byte stream. The CLI chooses it for `.jpg`/`.jpeg` output and does
not fall back to recompressing pixels if reconstruction data is absent. The
current capture supports 8-bit, full-canvas, single-frame/single-pass DCT8 JPEG
transcodes with original-color data, JPEG RAW quantization tables and compatible
CfL. Original progressive JPEG scan layouts are reconstructed from `jbrd`;
single-pass here refers to the JXL frame, not the original JPEG. Original RGB,
grayscale and supported YCbCr subsampling follow separate component mappings.
Pixel restoration filters do not change the captured quantized coefficients.

The maintained decoder currently still executes its normal decoding pipeline
while capturing coefficients. This is not yet a dedicated IDCT-free JPEG
reconstruction fast path. Extra DC precision, alternative quantization and
other unsupported reconstruction frame forms are errors, not fabricated output.
Unknown APP/COM records, restarts, scan refinement, padding and deduplicated
ICC/Exif/XMP are restored by the bounded inverse metadata parser and existing
JPEG entropy writer. All-one padding omitted by `jbrd` is reconstructed by its
wire default, with the same padding-work budget as explicit bits.

## Known Boundaries

- Native U31 integer output and independent raw-layer output are later APIs,
  beyond the initial rendered-pixel CLI target.
- Upstream mixed patch/color/extra-channel upsampling has an explicitly
  unsupported combination; inheriting the decoder does not remove it.
- Custom output CMS control, every djxl flag and every output file codec are
  not automatically implemented by providing the pixel API.
- This is a working decoder baseline, not full djxl parity. Runtime coverage
  and outstanding differences are recorded below, rather than inferred from
  successful compilation or upstream feature claims.

## Local Maintenance

Local patches expose complete extra-channel metadata, decode header names as
UTF-8, provide explicit legacy-profile compatibility and preserve F16
signaling-NaN payloads when widening to F32 and writing F16. The finite conversion
path retains upstream SIMD. Spline radius/clipping edge cases and arc-length
resampling now follow the C++ reference. JPEG reconstruction now captures integer
coefficients before dequantization and uses a bounded inverse `jbrd` parser;
it does not copy rendered pixels or replay a cached input JPEG. Keep these hooks
isolated from normal pixel decode, with their supported frame forms documented.

Do not run the vendor's full test suite as if its large upstream test corpus
were included: that corpus was intentionally excluded. Project integration
tests generate fixtures and verify public behavior. Optional independent
oracle checks exercise official `djxl` on the same streams.

## Verification Checkpoint: 2026-10-23

Host: Windows x86_64, Rust 1.97.1. The declared Rust 1.89.0 minimum also passed
`cargo check --locked --workspace --all-targets`. No Linux/macOS runtime or
decoder sanitizer/fuzzing acceptance is claimed. Optimized development checks
used `CARGO_PROFILE_RELEASE_LTO=false` and
`CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16` to reduce linking time; the project's
release profile was not changed.

Feature code was integrated before the consolidated workspace runtime suite.
That run passed the existing targets and 1,432 core unit tests, but exposed
incorrect raw-fixture assumptions and upstream UTF-8 name corruption in the new
decoder integration target. Both were corrected, then that target passed all
20 tests. One subsequently added legacy-level/resource-budget test also passed.
Only affected tests were rerun after corrections, not the entire workspace.
The five decoder CLI tests passed, covering 8/16-bit PNG with ICC, APNG frame
pixels/loop count/exact rational timing, selected-frame still output, NPY
layout/raw float words, PNM/PFM endianness and invalid option combinations.
Final workspace Clippy with warnings denied and project formatting checks pass.

Independent development oracles were djxl 0.12.0 (`a1ebdf2`) and 0.13.0
(`1e57a79`). Reports are local, ignored artifacts in `out/decoder-integration/`:

- `pixel-exact.json`: 12 lossless fixtures compared against both versions,
  all F32 sample words exact. Includes hidden RGB, Gray/GA/RGB/RGBA, U8/U16,
  all 65,536 U16 values, all 65,536 F16 patterns and F32 edge words.
- `jpeg-cross.json`: six original JPEGs encoded independently by Rust,
  cjxl 0.12 and cjxl 0.13. Rust restored all 18 original byte streams exactly,
  including grayscale/RGB/subsampling, progressive scans, restarts and metadata.
  This report is **not all green**: djxl 0.12 disagrees with the original on
  three cjxl 0.13 outputs, whereas Rust and djxl 0.13 are byte-exact. The
  targeted `jpeg-cross-old-decoder-investigation.json` records these separately.
  Rust-produced files passed reconstruction with both official versions.
- `official-pixel-corpus.json`: five of six supplied official JXL fixtures
  passed strict-profile decoding and a declared `1e-5` finite rendering
  tolerance. The four-frame cropped/blended animation was sample-word exact.
  `spline_on_first_frame.jxl` had maximum absolute error `1.222e-6`; the
  extended-size-container case differed by `1.777e-15`. Lossless fixtures were
  never checked with this tolerance.

The legacy `splines.jxl` needs `--allow-implicit-level10` because its complexity
exceeds the undeclared Level 5 limit. Enabling high precision alone left a
`7.4813e-4` maximum difference; changing the fixed-step resampler to the C++
residual arc-length algorithm reduced it to `2.2054e-6` (RMS `8.2150e-8`).
The final targeted `spline-resampling.json` passes both spline fixtures against
both official versions using the **unchanged** `1e-5` tolerance. The small
first-frame spline's final maximum error is `1.1325e-6`. This is numerical
rendering agreement, not bit-exact lossy decoding or a complete standards
conformance certificate. All six official pixel fixtures now pass their stated
criteria; only the legacy one requires the explicit level compatibility flag.

The vendor corpus remains outside the maintained source tree. An isolated
copy under ignored `out/decoder-vendor-validation/`, with fixtures from the
pinned clean checkout, allowed targeted vendor tests without adding images to
Git. The finalized ten spline-module tests (including both new boundary
regressions and arc-length sampling) and six JPEG coefficient-capture tests
passed there. This is not a claim that
the complete upstream suite was run.

## Verification Commands

During feature implementation use compile checks; execute the runtime matrix
after integrating the complete path:

```text
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace --release --no-fail-fast
```

`tests/decode.rs` generates in-memory pixel/frame/metadata fixtures.
`jpeg/reconstruction_decode_tests.rs` tests bounded parsing and full original-byte
round trips, not just intermediate models. `tests/decode_jpeg.rs` adds a bounded
real-image matrix when `RS_JXL_JPEG_INPUTS` is explicitly configured.
Ordinary tests do not require Python, C++ codecs, or downloaded image data.

The development-only `tools/verify_decoder.py` uses NumPy and independent
official CLI executables. All output goes into a temporary directory, removed
after testing. Optional JSON reports belong in ignored `out/`:

```text
python tools/verify_decoder.py --rs-jxl target/release/rs-jxl --djxl /path/to/djxl --jxl sample.jxl --jpeg original.jpg --cjxl /path/to/cjxl --report out/decoder-report.json
```

Repeat `--jxl`, `--jpeg`, `--djxl` or `--cjxl` for more cases/versions. The
default float comparison is exact word equality. An explicit nonzero `--atol`
permits and reports finite numerical differences for independently implemented
lossy rendering; nonfinite words must still match. JPEG reconstruction always
requires exact original bytes. This option never relaxes a lossless pixel test
silently, and compressed JXL byte identity is never an acceptance target.
