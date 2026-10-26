# Maintained Rust Decoder

## Acceptance Target

The first decoder milestone targets the practical capabilities of official
`djxl`: decode still images and visible animation frames, write integer/float
pixels, and reconstruct original JPEG bytes when reconstruction data is present.
It is not satisfied by parsing headers or by returning a re-encoded JPEG.
Full CLI flag parity and full format coverage are separate checks, not assumptions
made from adopting an upstream decoder.

### Lossless-First Scope Clarification (2026-10-24)

The user clarified that lossless functionality is the acceptance priority:
original sample equality (including invisible RGB), original JPEG byte equality,
and exact multiframe reconstruction. Native decode without color/format
conversion remains distinct from a requested rendered/color-converted output.

The five recorded VarDCT rendering differences of approximately `1e-5` to `2e-5`
are numerical diagnostics, not demonstrated lossless failures or blockers for
this milestone. Their existing reports and comparison thresholds remain intact;
they are not relabeled as passing. Do not continue changing rendering arithmetic
solely to satisfy that diagnostic threshold. Lossy decode must still handle the
file, dimensions, frames and channels correctly; crashes, nonfinite corruption
and substantial rendering errors remain defects to investigate.

These measurements concern our maintained decoder, not a measured pristine
upstream build. They neither establish an upstream bug nor certify full format
conformance. A pristine comparison was not completed before reprioritizing.

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

For repeated independent inputs, `Decoder` owns reusable worker resources:

```rust
use rs_jxl::decode::{DecodeOptions, Decoder};

let mut decoder = Decoder::new(DecodeOptions { threads: 4, ..Default::default() })?;
let first = decoder.decode(&first_jxl)?;
let selected = decoder.decode_frame(&second_jxl, 2)?;
let original_jpeg = decoder.reconstruct_jpeg(&jpeg_transcoded_jxl)?;
```

Only immutable options and the worker pool are retained. Each call creates fresh
parser, frame and reference state, including after a failed input. Resource limits
apply separately to each complete input. `options()` returns an immutable view;
create another context to change configuration. Calls require `&mut self`; this
is not a shared global decoder cache. At one thread no pool is created. The free
functions remain available and construct their own resources for each call.
Streaming already retains its resources for the lifetime of its input and is
unchanged. CLI `--num-reps` now reuses one context, so repeated-call benchmark
results must not be interpreted as equivalent single-image kernel speedups.
See [the measured experiments](CODEC_EXPERIMENTS_20261025.md).

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

`SampleFormat::Native` preserves source integer ranges. An integer declared as
10 bits is returned in U16 storage with values 0..1023. Wide original-color
Modular integers (17..31 bits) use a separate pre-float capture path and return
U32 storage, including extra planes. This path supports exact integer-compatible
composition, not arbitrary fractional rendering. Unsupported transforms,
resampling and mixed floating/integer capture return an error instead of rounded
samples; explicitly select F32 for rendered output. See
[native integer decoding](NATIVE_INTEGER_DECODE.md) for its precise boundaries.
Floating input is widened to F32. `U8` and `U16` explicitly request conversion
to their full storage ranges; `F32` requests normalized floating output, which
cannot preserve every U31 value.
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

`DecodeOptions::dither` selects the floating-point-to-U8 output pattern:
`DitherMode::JxlRs` (default) preserves upstream jxl-rs;
`DitherMode::Libjxl` uses libjxl v0.12.0's table for C++ output compatibility.
CLI: `--dither jxl-rs|libjxl`. Selection is per decoder, including previews and
extra output planes, not a process-global switch. Upstream jxl-rs deliberately
changed that table in [PR #940](https://github.com/libjxl/jxl-rs/pull/940) for
more even dithering; its different U8 samples were not evidence of damaged JPEG
coefficients. This output selection does not affect native integer lossless
samples, F32 output, Modular smooth-squeeze, or byte-exact original-JPEG
reconstruction. It does not
guarantee universal pixel equality: the upstream VarDCT floating-point pipeline
can still differ slightly from C++, even with identical quantizers. See the
bounded measurements in `JPEG_PIXEL_DECODE_20261025.md`.
Explicit CMS/background postprocessing retains its existing nondithered final
quantizer; the option selects the backend's already-dithered output.

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
RGB/XYZ clipping. The default now follows the tested skcms-based djxl conventions:
A2B0 then A2B1 selection independent of intent, unit-normalized table PCS,
and no intent-dependent absolute white-point adjustment on this route.

`ColorOptions::cmyk_icc_conventions = true` retains the previous intent-aware
table selection, legacy 16-bit Lab `0xff00` white encoding, XYZ scaling and
absolute media-white correction. The CLI exposes `--cmyk-icc-conventions` together
with `--color-space`. This is a scoped CMYK table policy, not a claim of complete
ICC conformance or a switch for every moxcms transform. Other ICC forms use
moxcms, with extended matrix handling and general-stage support enabled.

CMS output is not universally bit-interchangeable between engines. Neither
mode clips extended PCS/RGB output. Out-of-range or nonfinite device input still
fails explicitly instead of silently clamping. Native sample extraction and
original-JPEG reconstruction never use this optional color conversion.

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

- The 2026-10-24 strict floating-render matrix is 12/17 against each official
  decoder version, not an all-green conformance result. Five VarDCT cases exceed
  the unchanged `1e-5` diagnostic threshold; see the checkpoint below.
- Native U31 output has an explicit integer-only path; fractional blends,
  resampling, VarDCT and floating-channel mixtures are outside that exact path.
- Mixed-upsampled patch dictionaries are supported when color patches do not
  blend extra channels. Cross-resolution EC/alpha blending rejected by the C++
  reference is still rejected, rather than assigned invented semantics.
- There is deliberately no lossy JPEG output encoder or pixel-to-JPEG fallback.
  `.jpg` output always means original-byte reconstruction.
- Progressive selection currently consumes all frame sections while skipping
  eligible later HF entropy work. It is not a streaming/network-byte optimization.
  Required reference frames and unsupported progressions retain full quality.
- Incremental pixel input is a separate `StreamDecoder` / `decode_reader` API,
  described in [streaming input](STREAMING_INPUT.md). It retains parser state and
  can emit frames before EOF; callers must still validate `finish()`. Backend
  buffering and references prevent a constant-memory promise. JPEG restoration
  and metadata-only entry points still take complete byte slices.
- Output CMS is explicit; opaque ICC transforms reject out-of-range input instead
  of silently clipping it. Structured profiles use an unclamped matrix path.
- Numerical rendering and general ICC-engine differences with skcms-based djxl
  remain separate from native-lossless sample correctness. The default direct
  CMYK table route now follows its intent/PCS conventions; the explicitly
  selectable intent-aware route deliberately uses different conventions.
- Absolute ICC conversions with different media whites use explicit PCS stages.
  If the scaled intermediate exceeds the supported opaque-ICC numeric domain,
  conversion fails without clipping or publishing partial output. This is not
  a claim of arbitrary extended-range ICC coverage.
- Feature implementation does not imply complete standards conformance. Runtime
  coverage and remaining compatibility differences are recorded below.

## Error Categories

Backend sample/dimension ceilings and allocation failures retain `Error::Limit`
through pixel decoding, selected-frame decoding, inspection and JPEG restoration.
Backend I/O failures retain `Error::Io`; malformed bitstreams remain
`Error::Decoding`. Reconstruction's nested metadata, coefficient, entropy and
output budgets likewise retain their resource classification. These categories
do not turn unsupported files into accepted output or imply a whole-process
memory sandbox.

Only Exif/XMP boxes actually referenced by the JPEG reconstruction metadata must
be unique. Unrelated duplicate display metadata does not prevent restoring an
original JPEG. Missing or ambiguous required sources and duplicate `jbrd` boxes
remain errors, including compressed and trailing source boxes.

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

## Lossless Closure Checkpoint: 2026-10-24

After the lossless-first scope clarification, the affected release targets
passed 61 integration tests, 13 reconstruction/error unit tests and 35 CLI
tests. This was a targeted run, not another complete workspace or upstream
conformance suite. Checks include:

- Every integer value at every declared depth from 1 through 16, with one/four
  decoder threads. Floating special words, hidden RGB and extra-plane tests
  retain exact equality, never the lossy rendering tolerance.
- All eight orientations with negative/off-canvas crop origins, checking raw
  samples, oriented headers, selected layers, inspection and composed output.
- Automatic zero-duration reference chains, extra-only changes, actual mask
  winners with unequal color/alpha depths, and exact work/output-budget edges.
- JPEG source metadata before/after coefficient data, Brotli and EOF-sized
  boxes, missing/ambiguous required sources, unrelated duplicate display boxes,
  every incomplete synthetic container prefix, and typed resource failures.

The automatic-frame independent matrix passed 240 exact-sample comparisons
against official djxl 0.12 and 0.13, with 50 actual mask and 100 multiple-rectangle
winners; see [automatic frames](MULTIFRAME_AUTO.md). Two real JPEGs encoded by
Rust, cjxl 0.12 and cjxl 0.13 passed all six original-byte reconstruction cases
with both Rust and djxl 0.13. A separate check verified both djxl versions
restore the Rust-produced files exactly. Reports are ignored local artifacts
`lossless-final-jpeg-cross.json` and `lossless-final-jpeg-encoder.json` under
`out/decoder-integration/`. Previously documented old-decoder/new-encoder JPEG
differences are not relabeled or hidden by this scope.

The normal release CLI was rebuilt. Strict workspace all-target Clippy,
Rust 1.89 all-target compilation, formatting and whitespace checks passed.
No lossy-render arithmetic, performance policy, native-U31 extension or CI
changes were made in this closure pass. Runtime checks remain Windows x86_64;
this is not Linux/macOS, fuzzing, full decoder conformance or production
certification. Existing native-U31/CMS/streaming boundaries remain explicit.

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

## Compatibility Checkpoint: 2026-10-24

The integrated release workspace run passed 1,679 ordinary tests and one doctest.
Subsequent CMS corrections passed all 61 affected color tests in debug; 60 had
already passed in release before the new gamma fixture's cross-SIMD assertion
was corrected to its explicit color-conversion criterion. Native lossless tests
still compare exact samples/words. This does not substitute a tolerance for
lossless round trips. The 29 affected noise/upsampling/Gaborish/LF tests and 17
XYB-stage tests passed separately in the fixture-enabled vendor validation copy.
Only affected paths were rerun after fixes, not another full workspace matrix.
Final strict all-target Clippy, formatting and Rust 1.89 all-target compilation
passed. Runtime verification here is Windows x86_64, not Linux/macOS execution.

The default CMYK path now matches the local skcms conventions in A2B priority,
Lab/XYZ normalization, CLUT interpolation order and inverse transfer generation.
The latter uses skcms's approximate scalar `powf_`, not system `powf`. Structured
sRGB/gamma-to-linear also follows the ICC parameter quantization and transfer
approximation. The fixture named `3x3_srgb_lossless.jxl` actually declares a
gamma curve, so handling only the sRGB transfer enum did not cover it.

Renderer corrections use base rather than LF noise correlation, exact encoded
noise LUT precision, source-order filtering and the C++ Gamma/DCI near-black
cutoff in XYB output. None changes the lossless encoder, SIMD target policy,
CI, or original-JPEG reconstruction contract.

Final reports are ignored artifacts `out/decoder-integration/compatibility-final-012.json`
and `compatibility-final-013.json`. Both contain **12 passes out of 17**, using
the original `1e-5` finite-render threshold; native CMYK raw layers use zero
tolerance. The CMYK RGB conversion, gamma-to-linear fixture, PQ/HLG display and
unsubsampled extra-channel fixtures have zero differing sample words. No general
CMS or standards-conformance certificate follows from these fixtures.

The five outstanding diagnostics are identical against both official versions:

| Case | Maximum Absolute Difference |
| --- | ---: |
| 75%-length partial `progressive_ac` | 1.28825195e-5 |
| Progressive quality 2 | 1.28825195e-5 |
| Progressive quality 4 | 1.07800588e-5 |
| Progressive quality 8 | 1.01486221e-5 |
| `ec_upsampling8_multi_group` | 1.95843168e-5 |

In the last fixture the alpha/extra samples are exact; the discrepancy is in
base RGB. Full-quality `progressive_ac` has maximum `1.19339675e-5` and RMS
`3.04987718e-7`. The extra-channel fixture's RMS is `4.51581641e-7`. All values
are normalized floating output. Small magnitude alone does not prove the cause
or harmlessness. A source-order XYB matrix experiment was not retained because
it did not explain the residual and worsened some diagnosed pixels. Further
attribution needs matched pre-XYB intermediates; these cases remain failures,
not silently waived passes. Aggressive performance changes were not started.

Automatic multiframe selection has separate exact-sample acceptance in
[MULTIFRAME_AUTO.md](MULTIFRAME_AUTO.md), rather than sharing these lossy-render
tolerances.

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

## Native Samples and Incremental Input Checkpoint (2026-10-24)

The public Native path now retains 17..31-bit original-color Modular integer
samples without f32 storage. Color and extra planes keep their declared code
values; unequal-depth interleaved alpha is only a display convenience, and its
separate plane is authoritative. Integer RCT/palette/squeeze routes, multigroup
execution, reference crops and exact sample limits are covered by targeted
tests. This is the documented integer subset, not all fractional render modes.
See [native integer decoding](NATIVE_INTEGER_DECODE.md).

`StreamDecoder` and the `decode_reader*` APIs keep one decoder and runner alive
across chunks. They publish frames before EOF but still require EOF validation
for the complete file. CLI `--stream-input` reads incrementally and retains
decoded frames until transactional output publication. JPEG reconstruction and
metadata-only/selected-frame input remain separate one-slice APIs. See
[streaming input](STREAMING_INPUT.md).

Additional fixes validate patch entropy final state, signed coordinates, extents
and full-width blend symbols. Version-1 out-of-order chunks decode sequentially
without exposing unrepresentable seek checkpoints; version 0 still rejects
out-of-order chunks. This does not change the encoder's ordered version-0 output.

JPEG DC refinement no longer requires an unused DC Huffman table. Source tables
and selectors are retained, and original JPEG bytes are reconstructed exactly.
The bounded compatibility probe is diagnostic, not a new arithmetic transcoder.
Arithmetic JPEG remains an implementation/mapping investigation, not a claimed
format prohibition; CMYK original-JPEG archival has a distinct coefficient
mapping limitation. See [JPEG compatibility](JPEG_COMPATIBILITY.md).

Windows x86_64 targeted results at this checkpoint:

- 92 integration tests passed: ordinary decoding, raw layers, hostile input,
  JPEG reconstruction/capture, lossless edges, 17 streaming tests, seven wide
  integer tests, three DC-refinement tests and automatic multiframe selection.
  The hostile child worker is deliberately ignored for ordinary discovery and
  invoked by four parent tests; it is not counted twice. Two optional
  external-corpus tests were not run.
- 295 JPEG-module unit tests passed, including 14 compatibility-probe tests;
  nine optional reference-codec tests were not run.
- 82 CLI tests passed; two optional external-codec tests were not run. This
  includes streaming input, no output publication on failure, and exact NPY
  integer widening.
- 11 vendor native-integer helper tests and 28 patch-module tests passed in the
  existing isolated vendor fixture copy. This is not the full upstream suite.
- 10 platform-harness tests passed, including failure/timeout accounting and
  the prohibition on automatic WSL invocation.
- Full workspace/all-target compilation with the already-installed Windows
  Rust 1.89.0 minimum toolchain passed. Final Clippy with `-D warnings` and
  workspace rustfmt checks also passed.

Release test builds used `CARGO_PROFILE_RELEASE_LTO=false` and
`CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16` to reduce development link time. Runtime
logs are under ignored `out/decoder-integration/`; no bitstream-hash equality
was used as codec acceptance. Pixel tests use original samples; JPEG tests use
the original entire JPEG file. This is bounded regression coverage, not an
exhaustive fuzzer, a process-memory sandbox, or full format certification.

Linux runtime acceptance was not performed. WSL preparation was stopped at the
user's instruction; it must not restart implicitly. macOS is outside the user's
current acceptance scope. Python bindings and CI were not changed.

## Verification Commands

For single-thread JPEG reconstruction and subsequent pixel-decoder changes,
see [Single-Thread and Pixel Decoder Follow-Up](DECODER_SINGLE_20261024.md).
This records measured gains and remaining official-decoder performance gaps,
not a claim of universal speed parity or full corpus conformance.

For the later JPEG reconstruction performance iteration, including parallel scan
writing, unchanged original-byte acceptance, and remaining single-thread gaps,
see [Parallel JPEG Reconstruction](DECODER_PARALLEL_20261024.md). Its normal
Release measurements are separate from the reduced-LTO feature-test builds above.

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
