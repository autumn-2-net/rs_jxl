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
return. By default invisible zero-duration layers are composited internally.
Set `coalescing = false` to return every regular crop layer, including zero-duration
layers; indices then address layers, not visible frames. Internal reference-only
and LF frames remain dependencies, not public image layers. Each result keeps its
`header`, including crop origin, dimensions, color/extra blending, reference slot,
duration, timecode and preview flag. Raw layers are unblended source layers;
clients must use their headers rather than pretending they are full canvases.
Raw layers cannot request CMS/background conversion or a different alpha
association: independently converting ADD/BLEND layers would invalidate their
retained reference/blending instructions. Coalesce first, then transform.
Raw output containing spot channels also requires `render_spot_colors = false`
(CLI `--norender-spotcolors`); spot rendering would similarly invalidate layer
instructions. The unrendered spot planes remain available separately.

`include_preview = true` returns an embedded preview separately in `image.preview`.
It does not change frame indices. `inspect` scans frame headers, profiles and boxes
without decoding compressed pixel sections or allocating output canvases. Such a
scan validates the container/header structure, not the skipped pixel payloads.

`SampleFormat::Native` preserves source integer ranges through 16 bits. An
integer declared as 10 bits is returned in U16 storage with values 0..1023.
Floating input is widened to F32. Integers wider than 16 bits are normalized
F32, which **cannot preserve every U31 integer**. `U8` and `U16` explicitly
request conversion to their full storage ranges; `F32` requests float output.
`U8Bits(1..=8)` and `U16Bits(1..=16)` request an explicit significant-bit range.
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
When CMS is active, both primary-alpha representations pass through the same
final quantizer. This avoids a dithered planar alpha disagreeing with a rounded
interleaved alpha during explicit float-to-integer conversion.

Orientation is applied by default. Set `adjust_orientation = false` to keep
codestream order. Alpha association is kept as encoded; invisible RGB is not
cleared. Explicit `alpha_mode` conversion is available; it changes pixel semantics
and is not an exact archival roundtrip. High-precision rendering is enabled to match libjxl's default
`JXL_HIGH_PRECISION=1`, unlike upstream Rust's default low-precision spline
truncation. Spot-color rendering defaults on as in djxl; disable
`render_spot_colors` for the unmodified base color planes. Embedded and output
color profiles are distinct fields and must not be confused after color
conversion. CMYK is not represented by the project's RGB-only PixelLayout;
the Black plane remains available as an extra channel. An explicit RGB/gray CMS
destination can convert a CMYK source using its Black plane.

## Color and Partial Output

`DecodeOptions::color` contains `ColorOptions`: an optional `ColorProfile::Icc`
byte vector or structured `ColorProfile::Encoding`, `display_nits`, and an optional
background RGB triple. `ColorProfile::parse` accepts djxl descriptors such as
`RGB_D65_SRG_Rel_SRG` and `RGB_D65_SRG_Rel_Lin`; `parse_background` accepts black,
white or an RGB hexadecimal color. Arbitrary ICC transforms use the pure-Rust
`moxcms` engine; structured transfer/color conversions use explicit color math.
Color processing runs on float samples before requested quantization. It is
default-off and does not alter ordinary native-lossless output. Source metadata
remains source metadata: `info.bit_depth` does not describe a requested conversion.
`image.output_intensity_target` records the resolved display peak separately from
source tone mapping. EXR `whiteLuminance` and the JSON output field use that value;
an ignored SDR `display_nits` request does not relabel unchanged pixels.

Legacy CMYK `mft1`/`mft2` ICC tables are evaluated directly for structured RGB
or gray destinations: input curves, original four-dimensional CLUT, output curves,
then PCS conversion. This avoids a second fixed-grid resampling and premature
RGB/XYZ clipping. Absolute colorimetric intent applies the media-white XYZ ratio
after PCS conversion, preserving luminance. Other ICC forms use moxcms, with
extended matrix handling and its general-stage support enabled.

CMS output is not bit-for-bit interchangeable between engines. In particular,
the tested skcms-based djxl chooses A2B0 preferentially even for a requested
relative/absolute intent, and normalizes legacy 16-bit Lab tables differently.
This implementation retains the requested intent and the legacy Lab `0xff00`
white encoding. A full file can decode losslessly while an explicitly requested
color-space conversion produces different colors from another CMS. Native
sample extraction and original-JPEG reconstruction never use this conversion.

`allow_partial = true` permits flushed output at EOF. It does not convert corrupt
stream errors into success. `image.complete` and `frame.complete` distinguish
partial output; a file with no usable image pixels is still an error. Partial and
selected-frame calls do not claim to have validated trailing metadata.

`downsampling` accepts 1, 2, 4 or 8 and selects progressive quality at the original
canvas size, as djxl does. It is not a resize operation. Reference dependencies
must remain valid, so unsupported early-output cases decode at full quality.
`bytes_read` includes parser read-ahead and is not a claim about minimum network
transfer. JPEG reconstruction always requests all coefficient passes.

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
rs-jxl decode layers.jxl layer.npy --no-coalescing --metadata-out layers.json
rs-jxl decode image.jxl image.png --preview-out preview.png --orig-icc-out source.icc
rs-jxl decode image.jxl image.exr --alpha-mode premultiplied
rs-jxl decode image.jxl image.pam --bits 12
rs-jxl decode image.jxl metadata.xmp
rs-jxl decode image.jxl --disable-output --num-reps 5 --print-read-bytes
```

PNG writes visible animation frames as APNG when the timebase is exactly
representable; it does not silently discard other frames. NPY stores main color
channels followed by each extra channel once, including alpha. PPM/PGM/PFM
require one compatible frame; they cannot silently discard alpha/extra planes.
The CLI rejects unsupported sample/metadata representations rather than
implicitly quantizing float data. File output is published transactionally;
existing files require `--force` and cannot alias the input.

The CLI is split into controls, metadata, output-set planning and individual
writer modules. PAM/PNM/PGX and OpenEXR complement PNG, NPY and PFM. Portable
integer outputs support explicit 1..16 significant bits. EXR is F32 linear sRGB;
alpha premultiplication must be explicit when required. `--output-frames` and
`--output-extra-channels` retain frame/channel data in separate numbered files.
Raw-layer output requires `--metadata-out` to keep crop/blending/reference data.
PNG background compositing is explicitly requested with `--alpha-blend`; the
original alpha can still be exported with `--output-extra-channels`.

Metadata-only `.exif`, `.xmp`/`.xml`, `.jumb`/`.jumbf` outputs use header/box scans,
not full pixel decoding. `--icc-out`, `--orig-icc-out` and `--metadata-out` expose
output profile, source profile and JSON frame/channel metadata independently.
`--disable-output`, `--num-reps`, `--quiet` and `--print-read-bytes` support
measurement without writing decoded raster files. djxl underscore-style aliases
are accepted for corresponding controls. Multiple files are serialized and
preflighted before publication; each file is transactional, but the entire set
is not a filesystem-wide atomic transaction.

`decode::reconstruct_jpeg(&bytes, &DecodeOptions::default())` returns the
original JPEG byte stream. The CLI chooses it for `.jpg`/`.jpeg` output and does
not fall back to recompressing pixels if reconstruction data is absent. The
current capture supports 8-bit, full-canvas, single-frame DCT8 JPEG
transcodes with original-color data, JPEG RAW quantization tables and compatible
CfL. Original progressive JPEG scan layouts are reconstructed from `jbrd`;
multi-pass JXL coefficients are accumulated before capture is completed. Original RGB,
grayscale and supported YCbCr subsampling follow separate component mappings.
Pixel restoration filters do not change the captured quantized coefficients.

The coefficient-only path skips pixel/IDCT pipeline construction and LF smoothing.
Extra DC precision is accepted when integer JPEG coefficients can be recovered
exactly. Compatible non-unit global, DC and block quantizers are supported.
Fractional DC, clamping/overflow, non-RAW/non-JPEG-denominator tables and other
unsupported reconstruction frame forms are errors, not fabricated output.
Unknown APP/COM records, restarts, scan refinement, padding and deduplicated
ICC/Exif/XMP are restored by the bounded inverse metadata parser and existing
JPEG entropy writer. All-one padding omitted by `jbrd` is reconstructed by its
wire default, with the same padding-work budget as explicit bits.

## Known Boundaries

- Exact native U31 integer output remains beyond the rendered-pixel djxl target.
- Mixed-upsampled patch dictionaries are supported when color patches do not
  blend extra channels. Cross-resolution EC/alpha blending rejected by the C++
  reference is still rejected, rather than assigned invented semantics.
- There is deliberately no lossy JPEG output encoder or pixel-to-JPEG fallback.
  `.jpg` output always means original-byte reconstruction.
- Progressive selection currently consumes all frame sections while skipping
  eligible later HF entropy work. It is not a streaming/network-byte optimization.
  Required reference frames and unsupported progressions retain full quality.
- Output CMS is explicit; opaque ICC transforms reject out-of-range input instead
  of silently clipping it. Structured profiles use an unclamped matrix path.
- Numerical rendering and ICC policy differences with the tested skcms-based
  djxl remain. In particular CMYK-to-RGB is not claimed to match that CMS's
  intent/Lab-normalization behavior; raw CMYK layer samples do match exactly.
- Absolute ICC conversions with different media whites use explicit PCS stages.
  If the scaled intermediate exceeds the supported opaque-ICC numeric domain,
  conversion fails without clipping or publishing partial output. This is not
  a claim of arbitrary extended-range ICC coverage.
- Feature implementation does not imply complete standards conformance. Runtime
  coverage and remaining compatibility differences are recorded below.

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

## Feature Verification: 2026-10-24

The feature implementation was integrated before running the consolidated
workspace tests. That run passed 1,462 core unit tests (including 30 new color
tests), all 25 pixel-decoder integration tests, and 64 CLI tests. A new backend
scan test initially failed while constructing an invalid unsaved-reference fixture,
not while decoding. After correcting its reference slot, that targeted test
passed. The expanded pixel-decoder integration target subsequently passed all
29 tests, and the final CLI target passed 66 tests (two externally configured
tests ignored). These reruns cover alpha-copy consistency, raw-layer transform
guards, format-aware profile inspection, actual PNG/NPY frame splitting and EXR
output luminance. The full workspace suite was not repeatedly rerun after each
local fix.

The final targeted color/CMS run passed all 42 tests, including direct CMYK
tables, legacy Lab black/white, XYZ above one, input/output-curve ordering,
absolute media white (including two-stage opaque ICC), gray destinations,
retained alpha and transactional rejection of unsupported PCS ranges. Final workspace
Clippy (`-D warnings`), formatting, and Rust 1.89.0 all-target compile checks pass.
The release CLI was rebuilt after the final changes. A real-file CLI smoke
check also passed CMYK-to-gray with retained extra planes and PQ-to-EXR with
source peak 10,000 nits versus resolved output peak 250 nits.
These remain Windows x86_64 checks, not Linux/macOS runtime acceptance.

Independent exactness gates passed:

- `pixel-exact-extended.json`: 12 fixtures against djxl 0.12 and 0.13, with zero
  differing sample words, including all U16 values, all F16 patterns, special
  F32 words, hidden RGB and 8/16-bit Gray/GA/RGB/RGBA.
- `jpeg-coefficient-only.json`: six original JPEGs through Rust, cjxl 0.12 and
  cjxl 0.13; all 18 original-byte reconstructions exact. djxl 0.13 independently
  restored the same originals. The previous documented djxl 0.12 reconstruction
  incompatibility with some cjxl 0.13 outputs is not relabeled as a Rust failure.
- Ten maintained vendor JPEG-capture tests pass in an isolated fixture-enabled
  copy, including multipass accumulation, extra DC precision, quantizer checks,
  no-IDCT scratch use and transactional finalization. This is unit coverage;
  the real-JPEG matrix does not claim every synthetic coefficient form exists
  in an independent encoder fixture.
- Raw CMYK: four differently sized layers, native 8-bit C/M/Y/K/alpha exact
  against official PAM. Raw RGBA: five layers compared against official PFM and
  separate alpha, maximum RGB error `2.027e-6`, alpha exact.

Official NPY rejects unequal-sized raw layers before its frame-splitting stage.
The maintained feature oracle therefore uses PAM/PFM for raw-layer comparisons,
not a fabricated success or omission of those layers. The CMYK fixture needs an
explicit CMYAK-to-CMYKA channel reorder because the two serializers choose
different alpha/extra-channel ordering.

Progressive/partial rendering was also compared with full rendering. Full quality
on `progressive_ac.jxl` differs from djxl by at most `1.1934e-5` (RMS `3.057e-7`);
ratios 2/4/8 and a 75%-length partial input range from `1.0780e-5` to `1.2883e-5`
(RMS `2.422e-7` to `3.013e-7`). Deliberately comparing the wrong progression gives
RMS errors `0.011` to `0.041`. All outputs keep the original canvas size. These
small existing float-render differences exceed the initial `1e-5` diagnostic
threshold even at full quality; they are not pixel-word equality claims.

The targeted final preview/layer/extra-channel matrix passed all seven selected
cases. ICC gray conversion has maximum error `7.179e-6`; format-only inspection
returns exactly the same ICC as pixel output at both 8 and 32 bits. PQ and HLG
display fixtures are sample-word exact after the PQ approximation and luminance
rounding corrections. Structured sRGB-to-linear uses analytic transfer math;
its `1.395e-5` difference from skcms's approximate power calculation is retained
rather than reducing precision solely to match another CMS.

CMYK diagnostics are deliberately **not labeled oracle-parity passes**. The
direct-table path preserves negative RGB and XYZ above one, applies legacy Lab
scaling, and honors destination intent. The skcms fixture comparison still
differs (perceptual maximum about `0.0494`, relative about `0.2357`), including
the expected legacy-Lab white-point offset. A separate installed LittleCMS
development oracle confirms the absolute-whitepoint correction; its own
interpolation and negative-transfer rules also differ. None of these numbers
is a lossless sample tolerance or a claim of complete CMS conformance.

Reports belong in ignored `out/decoder-integration/`. Source fixtures and local
absolute paths are not part of the committed documentation or test inputs.

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

`tools/verify_decoder_features.py` tests preview, raw layers, color/display
processing, progressive selection, partial files and extra-channel fixtures from
the pinned upstream `resources/test` directory. Its tolerance and any deviations
are explicit in each JSON row. Use `--case` for targeted reruns after a correction:

```text
python tools/verify_decoder_features.py --rs-jxl target/release/rs-jxl --djxl /path/to/djxl --corpus /path/to/jxl-rs/jxl/resources/test --case preview --report out/preview-report.json
```
