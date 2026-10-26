# Pixel Acceptance Audit

Historical audit: the default resource ceilings listed below were removed on
2026-10-26. Current limits and verification are in
[RESOURCE_LIMIT_AUDIT.md](RESOURCE_LIMIT_AUDIT.md) and [PIXEL_API.md](PIXEL_API.md).

2026-10-23. **Complete pixel-lossless acceptance remains open.** This is a
source audit and focused regression handoff, not a complete-acceptance declaration.
The read-only C++ checkout HEAD was verified as
`b866b3defec13494bfbaf705db93581cda118e37`. Its optimized algorithms and public
API are not interchangeable notions of the supported sample domain.

Ownership is limited to this document,
`crates/rs-jxl/tests/pixel_acceptance.rs`, the subsequently authorized
`crates/rs-jxl/tests/metadata_fields_reference.rs`, and `tools/verify_port_corpus.py`.
The root is a virtual Cargo workspace, so the new `tests/pixel_acceptance.rs`
belongs under the library crate. No production changes or commits were made.
The parent owns centralized test runs. Prior 741-core/10-API/35-CLI and all
48 official suites are the supplied checkpoint, not rerun results from this audit.
Latest parent-reported workspace checkpoint: **813 core + 14 acceptance tests
passed** after the three production fixes. This does not establish a new official
decoder result. The metadata gate and final centralized run remain pending.

## Findings And Reproducers

**PA-01, PA-02 and PA-03 are FIXED, pending final centralized validation.** Their
corrections were re-inspected below; historical descriptions and line references
retain the original reproducers. The Rust tests are not compiled or executed by
this audit; the latest passing acceptance result is parent-reported above.
None of these findings demonstrated a corrupted pixel stream.

### PA-03: Late Legacy MA Sample Budget And 32-Bit Overflow (P2)

**FIXED, pending final run.** `legacy_sample_count` checks the population sum;
`collect_samples` rejects excess target samples before reserving or visiting
pixels, and `legacy_midpoint` now computes in u128. `training::gather_legacy`
checks remaining local allowance and consumes shared allowance before calling
the collector. The parent/MA owners made these changes. No 32-bit execution is
claimed by this audit.

Original behavior: `modular/encode.rs:412` chose `min(tree_samples, population)`
and immediately allocated that entire Sample vector. The visitor created every selected record,
including its property/residual vectors. Only after `collect_samples` returns
does `modular/training.rs:281` compare the batch against `sample_limit` and the
shared learner budget. Thus the declared hard selected-sample ceiling is not a
pre-allocation/traversal ceiling on this path.

Small repro: Gray U8 `[0, 1, ..., 63]`, Zero predictor, no fixed tree,
`tree_samples=64`, `max_tree_leaves=1`, `sample_limit=1`. Sixty-four records are
built before `Error::Limit`; a limit of 64 succeeds. The new
`legacy_sample_limit_rejects_instead_of_silently_truncating` test checks the
external error and untouched sink, **not allocation accounting**. The allocation
ordering is source evidence. At the allowed million-record target and 60
previous-channel properties, property vectors alone can reach 1 GiB before
rejecting `sample_limit=1`. Do not run that allocation-sized repro.

The same collector computes `(2 * next + 1) * count / (2 * target)` in `usize`
at `modular/encode.rs:425`. On a 32-bit target, a 256x256 Gray image with
`tree_samples=65536` reaches `65537 * 65536` at `next=32768`: debug overflow
instead of a resource error; release wraps the sampling-position calculation.
The new 32-bit-only regression uses just 64 KiB of source samples and
`sample_limit=1`. The fix addresses both budget ordering and midpoint arithmetic;
the external acceptance test alone does not measure allocation ordering.

### PA-01: Difference Planning Compares Float Storage, Not Declared Words (P2)

**FIXED, pending final run.** Color and planar-extra comparisons in
`multiframe::difference_bounds` now use `Image::modular_sample`, comparing exact
declared-format words without integer-to-float conversion. The matching
regression is included in the parent's reported 14 acceptance passes.

Original behavior: at `multiframe.rs:65` and `multiframe.rs:84`, comparisons called `PixelData::get`,
which returns the container word. But `Image::with_metadata` allows exact F32
storage for declared binary16, and `compatible` does not require matching
storage variants. This is a valid input pair:

```rust
let half = Image::new(2, 1, PixelLayout::Gray,
    PixelData::F16(vec![0x3c00, 0x8000]))?;
let single = Image::new(2, 1, PixelLayout::Gray,
    PixelData::F32(vec![1.0, -0.0]))?
    .with_metadata(half.metadata().clone(), vec![])?;
assert_eq!(multiframe::difference_bounds(&half, &single)?, None);
```

The audited pre-fix source returns the full 2x1 rectangle: `0x3c00 != 0x3f800000` and
`0x8000 != 0x80000000`. Planar extras have the same bug. It produces false
changes and unnecessary crop/reference work, not demonstrated sample loss.
Regression: `equivalent_f16_and_f32_storage_has_no_difference_bounds`.
The required contract remains exact declared-format comparison, preserving
signed zero and NaN payload distinctions rather than numeric floating equality.

### PA-02: Ineligible Depth Bypasses Malformed Patch Limits (P2)

**FIXED, pending final run.** Both `patch_frame` and `source_patch_frame` validate
their requested configuration before sample-depth applicability gates. The
matching regression is included in the parent's reported acceptance passes.

Original behavior: `patch_frame.rs:26` returned `Ok(None)` for floating or >16-bit color before
`PatchOptions::validate` is called in discovery. `source_patch_frame.rs:42`
does the analogous early return before `SourcePatchConfig::validate`.

Repro: encode a 1x1 Gray image with
`patches=Some(PatchOptions { max_patch_size: 0, ..Default::default() })`.
U8 `[1]` is rejected, but U32 `[1]` (declared U31) or F32 `[1.0]` is accepted
and produces output. The same malformed nested limits bypass validation through
`source_patches`. Regression:
`malformed_patch_limits_are_rejected_even_when_sample_depth_is_ineligible`.

This is inconsistent public configuration validation, not a request to force
patches on unsupported samples. Valid heuristic patch requests may legitimately
produce no dictionary. Validate malformed options before applicability gates.

## Precise Acceptance Contract

These are the inspected current domains and required acceptance criteria, not
a proposed exhaustive combination matrix. API construction, successful encoding,
and independently recovered output are separate gates.

| Surface | Current Domain / Required Contract | Evidence Or Remaining Gate |
| --- | --- | --- |
| Integer storage | U8/U16/U32 with declared unsigned depth 1..31, no rescaling, every sample within declared range, depth no larger than storage; U32 defaults to 31 | Existing depth tests; new integer/float rejection test and U31 adjacent-value planner test |
| C++ integer comparison | `image_metadata.cc:60` admits <=31; `encode.cc:632` public `CheckValidBitdepth` admits only 1..24 | U25..U31 is not parity with the frozen C++ public input API; independent full-bit recovery remains open |
| Floating storage | F16 words require 16/5 metadata; F32 supports exponent 2..8 and mantissa 2..23, total = sign + exponent + mantissa | Custom F32 must survive source conversion/inverse bit-exactly, not merely be numerically representable; existing float tests already cover this and are not duplicated |
| Float identity | Preserve signed zero, infinities and NaN payloads on supported paths | Existing special-word oracle fixtures; PA-01 declared-word planner fix is in place, pending final run |
| Color/layout | Gray, GrayAlpha, RGB, RGBA; interleaved channel count fixed; no color conversion, EXIF rotation, alpha cleanup or premultiplication | Existing layout/hidden-RGB oracles; RGB-like CMYK representation requires a CMYK ICC and exactly one Black extra, not a fifth PixelLayout |
| Extra planes | Alpha, Depth, SpotColor, SelectionMask, Black, CFA, Thermal, Optional; each has its own depth/storage; interleaved alpha is first and full-resolution | Planar sample count = ceil(width/2^shift) * ceil(height/2^shift), shift 0..3; new hostile shift/mixed-depth test |
| Extra counts | Metadata accepts <=4096; encoder Level 10 accepts <=256, in addition to color channels | New 1-pixel 256/257 boundary test; constructing an Image is not encoder acceptance |
| Geometry | Image constructor checks nonzero dimensions, checked product and exact buffer length; encode caps each axis at 262144 | New boundary preflight test; source Level 10 allows axes <=2^30 and area <=2^40 (`encode.cc:561`), so Rust is intentionally narrower |
| Frame canvas | Supplies global dimensions/metadata; samples are not encoded unless explicitly listed; canvas still passes encoder limits | New unencoded-canvas/reference-image resource test |
| Frame metadata | Color, depth, layout, extras including names/order/association, orientation, tone mapping and buffer declaration must match canvas | Animation is global: `encode_frames` overwrites the local metadata copy's animation before comparison. `plan` instead requires identical full metadata. New extra-name/tone mismatch tests |
| Layers/references | Zero duration is a layer. Nonzero duration requires animation. Slots 0..3; zero-duration slot 0 is saved, nonzero-duration slot 0 is not | Existing focused stage/overwrite/slot-zero tests already cover this; no duplicate matrix |
| Reference-only | Must precede visible output; no duration/timecode/origin/blending. Post-transform reference must cover canvas | Pre-transform slots cannot serve color OR extra blend backgrounds; patches reserve slot 3. Existing oracle/reference tests plus new independent extra-source check |
| Crop/composition | REPLACE preserves arbitrary hidden RGB. Arithmetic/alpha modes implement requested blending, not original-frame identity | Each extra has its own blend source; empty extra_blend inherits color blend. Explicit sample crops must align extra grids. Subsampled auto plans retain full frames because upsampling precedes composition |
| Frame fields | Names <=1071 UTF-8 bytes, not characters. Duration/timecode/loops are u32. Timecodes require enabled animation timecodes | Timebase numerator 1..2^30, denominator 1..1024. Boundary tests verify API acceptance; the new ignored metadata test checks official retrieval on one modest sequence, not raw tick endpoints |
| Origins | i32 API type, but signed packing into the header narrows the representable range to -536880256..536880255 | Parent moved rejection into `FrameOptions::validate`, using shared `FRAME_SIZE_CHOICES`: out-of-range values return InvalidInput before pixel encoding. Endpoint unit coverage and i32-extreme acceptance checks remain subject to final centralized validation |
| Metadata floats | Spot RGBA and tone mapping use source truncating F16 conversion; finite/range constraints apply | Metadata f32 inputs are NOT promised bit-identical f32 metadata on output; this is distinct from floating pixel losslessness |
| ICC/levels | Structural RGB/GRAY/CMYK ICC guard, not full CMS validation; encoder ICC <=2^28 bytes | Default 32-bit Modular scratch declaration requires Level 10. Level 5 additionally needs <=4 extras, ICC <=2^22, no Black, allowed geometry and a truthful 16-bit scratch promise |
| Scratch promise | `modular_16_bit_buffer_sufficient=true` rejects floating and >12-bit channels at encode | <=12 bits is necessary, not proof every chosen transform intermediate fits. This remains a caller promise, not an independent analysis |
| Output API | Complete stream buffered before first Write; returned total including container <=max_output_bytes | New aggregate multiframe inclusive-limit, late-invalid-frame, Interrupted/WriteZero/partial-I/O tests; I/O failure can leave a partial sink |
| Invalid options | Malformed requested settings must fail independent of image eligibility | PA-02 fixed, acceptance regression reported passed, final run pending; valid optional tool gates remain separate from malformed settings |

## Resource Boundaries

- `threads`: 1..256; default 1; validation precedes creation of a per-call pool.
  Serial execution and caller-pool reuse already have dedicated coverage.
- `max_pixels`: default 64,000,000, per canvas/frame image. It is not total samples
  including extras, sum of animation areas, number of frames, CPU time, or RSS.
  Three one-pixel layers are accepted with `max_pixels=1`; the new test records
  that scope explicitly rather than assuming a sequence-wide ceiling.
- `max_output_bytes`: default 512 MiB, hard bound on a successful returned stream,
  not peak allocations. Groups/sections are encoded before total checks; a frame
  is appended before the sequence-size check, and the container is assembled
  separately. A tiny budget is not a universal pre-work/memory guard.
- `tree_samples`: 1..2^20; `max_tree_leaves`: 1..16384;
  `sample_limit`: positive, default 2^20, a ceiling rather than a sample target.
  Shared learners share a ceiling; unrelated local groups need not share one
  image-global ceiling. PA-03's legacy pre-consumption checks are now in place. Gather scratch
  byte estimates are not process-RSS limits.
- Group shift 0..3; previous channels <=60; <=30 explicit squeeze passes;
  palette limit 1..70000; RCT IDs <42, search candidates 1..19; weighted preset
  0..4, search candidates 1..5. Existing tool-validation tests cover these.
- Caller-owned Image buffers exist before encoder limits are consulted.
  Metadata clones, plain Vec allocations, frame lists and intermediate planes
  prevent an unconditional recoverable-OOM guarantee. There is no public
  aggregate frame/sample/scratch/time budget or cancellation contract.

`docs/PIXEL_API.md` now documents these local ceilings and their scope, and PA-03
is fixed pending final validation. These are not an untrusted-input memory sandbox.
No giant allocations or deliberately exhausting thread tests were run here.

## What Official Decoding Actually Proves

The supplied all-48 result remains useful evidence, with these boundaries:

| Existing Target | Actual Assertion / Limit |
| --- | --- |
| `reference.rs` (15 oracle suites) | Exact decoded PNG U8/U16 samples across layouts/tools/source paths, hidden RGB, oriented output, ICC byte extraction and extra planes; APNG checks displayed pixels/count |
| `float_reference.rs` (16) | NPY f32 words for float/custom/special-value fixtures; U17..U24 integer recovery on tested fixtures; U31 checks projection only |
| `metadata_frame_reference.rs` (3) | Mixed depths, patches, and nonconstant subsampled multiframe plans; compare displayed original colors and decoded upsampled extra planes against independently described full-frame baselines |
| Integer/floating Add (2 + 1) | Active dictionaries and typed exact-replay guards, including mixed extras; this is not arbitrary arithmetic blend recovery |
| Scheduler frontier (11) | Official recovery for connected scheduler outputs; not proof of source speed/size parity or of every untested routing condition |

**U25..U31 exact original integer recovery is not established.** In particular,
binary32 NPY collisions cannot be repaired by epsilon comparisons, larger PNG
output, or calling a float projection an integer oracle. The frozen external
integer output conversion is <=16 bits (`dec_external_image.cc:280`), and its
public pixel types expose UINT8/UINT16/FLOAT/FLOAT16, not a full U31 output.
The new test keeps distinct adjacent U31 values in the Rust planner without
pretending a successful encode proves those bits survived an official decoder.
Close this gate with independently observed pre-normalization integer samples,
or explicitly narrow the accepted high-depth claim. Neither happened here.

**Metadata acceptance is narrower than pixel acceptance.** The inspected
`reference.rs:575` APNG helper does not assert delay numerator/denominator;
existing fixtures set names/timecodes but do not retrieve and compare them.
NPY payload comparisons do not verify extra names, CFA index, spot RGBA,
association flags, timebase/loops, or zero-duration/reference-only raw headers.
`djxl --metadata_out` can check visible frame name/timecode and projected duration,
but `lib/extras/enc/npy.cc:145` serializes duration in floating seconds, not raw
ticks, and does not expose every raw-layer field. A pass there cannot prove all
u32 timing endpoints or reference-only metadata.

### New Independent Metadata Gate (Pending Central Execution)

`metadata_fields_reference.rs` adds one ignored test for **both** preserved
official `jxlinfo` tools. The source `tools/jxlinfo.cc:112` disables coalescing and
keeps orientation. Its verbose extra descriptors and animation/frame records
expose more than the djxl NPY metadata path. The inspected source at b866b3d and
the local v0.12.0 tag use the asserted text format; the preserved 0.13 executable
must still pass centrally. No executable version is inferred from its basename.

The bounded fixture is 4x3 RGBA8 with a named, unassociated Alpha8 and named
12-bit Depth plane at dim_shift=1. Three regular frames comprise a named
zero-duration layer saved in reference slot 1, a 2x2 REPLACE crop at (2,0) with
explicit color/extra source 1, and a final full frame. It asserts:

- The complete image summary, color/extra counts, both ordered extra descriptors
  (type, bit depth, name, alpha association, depth upsampling), and default
  intrinsic size/orientation/preview fields. Unexpected extra descriptor lines,
  including a floating exponent on an integer extra, fail exact block equality.
- Timebase **8 / 2**, loop count **3**, enabled timecodes; the complete ordered
  `Layer:` / `Frame:` list with all names, raw crop geometry and timecodes
  **0x01020304 / 0x01020305 / 0x01020306** (jxlinfo prints uppercase hex).
- Printed durations **0.0 / 250.0 / 500.0 ms** and loop-adjusted animation length
  **2.250 seconds**. The modest millisecond values are exactly representable;
  jxlinfo computes them in float and prints one decimal, so **raw u32 tick
  endpoints and arbitrary timebases remain unproved**.

The process check requires success, empty stderr, and complete expected records:
the inspected jxlinfo can otherwise return zero after a decoder error following
BASIC_INFO. Each invocation has a 30-second timeout and 64-KiB-per-log cap, with
kill/reap on failure; fixture/logs live only in a unique temporary directory.

This is **header inspection, not a new pixel decode**. jxlinfo does not print
save_as_reference, color/extra blend modes/sources, or save-before-transform.
Their exact wire values are not independently asserted here. The initial frame
is a regular zero-duration layer, **not `reference_only=true`**: with coalescing
disabled, `decode.cc:1350` exposes regular zero-duration frames, but does not
promise an event for reference-only frames. Those raw headers, arbitrary extra
types/float metadata, name/timing endpoints, and full U31 samples remain outside
this focused gate. Existing independent pixel/composition tests remain necessary.

## New Tests And Corpus Handoff

`pixel_acceptance.rs` has 14 tests on 64-bit targets and one additional 32-bit
regression. They target type separation, representational boundaries, independent
extra references, late error atomicity, sink failures, and the findings above.
No new decoder matrix is included. Custom-float representability, repeated
predictor/RCT combinations, slot-zero lifecycle and reference-stage overwrites
remain covered by their existing focused tests.
`metadata_fields_reference.rs` adds just the single ignored sequence above,
running the same encoded bytes through two independent tool versions.
Its exact-count scalar assertions and ordered extra/frame equality reject missing
or duplicate selected records. The metadata file is ready; **edits are frozen
for the parent's centralized tests**. No external metadata pass is claimed.

The optional corpus harness uses eight explicit originals, header-preflighted
at **2,478,708 total pixels**, with no recursive scan or source edits:

| Case | Original Under jxl_testdata | Shape / Precision |
| --- | --- | --- |
| photo | `images/photo_kodak/kodim01.png` | 768x512 RGB8 |
| icon | `images/icon_64/actions-address-book-new.png` | 64x64 RGBA8 |
| texture | `images/textures_photo/clover.png` | 1024x1024 RGB8 |
| game-ui | `images/screenshot_game/20XX_Menu_picture.png` | 634x406 RGBA8 |
| photo16 | `testdata-main/external/wesaturate/64px/Nikon-D3-14bit_2020_g1.png` | 64x64 RGB16 |
| gray | `testdata-main/external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png` | 500x500 Gray8 |
| rgba | `testdata-main/external/wesaturate/500px/tmshre_riaphotographs_alpha.png` | 500x500 RGBA8 |
| gray-alpha | `testdata-main/jxl/flower/flower_small.ga.png` | 510x532 GA8 |

The harness reuses tested OpenCV/numpy PNG sample helpers from
`compare_cpp_port.py`, but never its C++ encoder or benchmark entry points.
It preserves source precision, PLTE/tRNS/IDAT, original hidden color values,
and rejects unsupported formats/depths rather than narrowing them. Metadata-free
temporary PNGs are explicit pixel-only inputs, not metadata-preservation evidence.
It reports exact mismatch coordinates, JXL size/bpp, one-shot CLI time/throughput,
decoder versions, actual hidden-RGB population and non-8-bit-representable U16
sample counts. No compressed-byte hashes are generated or compared.

Run limits: <=8 images, <=2.1 MP each, <=5 MP total, <=16 MiB original PNG,
<=32 MiB output file, <=1 MiB per child log, <=2 explicit official djxl binaries,
one thread setting (1..8), one mode, no repeat/warmup matrix, <=120 seconds per
child and <=600 seconds orchestration budget. Child limits kill/reap processes;
polling can briefly overshoot file limits. In-process PNG work is checked at
phase boundaries, not forcibly interrupted. These are not OS RSS guarantees.
Missing fixtures, failures and budget overruns fail the run; no silent skips.
Only `--list` was run here, not encodes, self-tests, or corpus verification.
Both new Rust targets pass `rustfmt --check`; Python `--help` parses/imports
successfully with `-B` (no bytecode files). Neither is a functional test result.

The corpus self-test now contains 13 focused orchestration tests, without real
child processes, image libraries, or access to originals. It covers source-root
containment/duplicates/input caps, 8/16-bit decoder arguments, bounded reads,
expired deadlines, stale output, kill/reap on timeout and file growth, fast-exit
status/output/log/deadline checks, and the manifest/header/mode limits. Review
found and fixed stale-output acceptance and a missing post-exit deadline check
in this new harness. Mocked tiny temporary files exercise these boundaries;
existing `test_compare_cpp_port.py` remains responsible for pixel-reader fidelity.

Parent commands, from repository root:

```powershell
cargo test --locked -p rs-jxl --test pixel_acceptance
cargo test --locked --release -p rs-jxl --test pixel_acceptance
$env:RS_JXL_JXLINFO = @(
  '../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/jxlinfo.exe'
  '../jxl_python/out/build/windows-clang-cl-libjxl013/libjxl/tools/jxlinfo.exe'
) -join [IO.Path]::PathSeparator
cargo test --locked --release -p rs-jxl --test metadata_fields_reference -- --ignored --nocapture
python -B tools/verify_port_corpus.py --self-test
python -B tools/verify_port_corpus.py --list
python -B tools/verify_port_corpus.py --encoder target/release/rs-jxl.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --decoder ../jxl_python/out/encoder_opt_baseline/djxl.exe
```

These are the parent's trusted paths; both listed jxlinfo files exist. The
0.13 jxlinfo is **not** in encoder_opt_baseline, which preserves only cjxl/djxl.
The jxlinfo CLI has no --version: attach companion cjxl/build provenance to
central results. Executable origin is caller-attested; a basename, directory
label, or version string alone does not establish an official build.
Default encoding is one Rust source candidate at effort 7, not a runtime C++
encoder and not full scheduler acceptance. Use `--case photo16 --case icon`
for a smaller check, `--mode legacy` for direct defaults, or an explicit
`--mode source-search --case photo --case icon` for a bounded scheduler check.
`--image` permits explicit original PNGs within read-only `test_img` or
`jxl_testdata` roots, still subject to all budgets. JPEG is intentionally excluded:
comparing another JPEG decoder's output to zune-jpeg would conflate decoder
differences with JXL loss. The JPEG input/sample contract needs separate evidence.

## Documentation And Completion Gates

1. **FIXED, final run pending:** PA-01/PA-02/PA-03. Parent reports 813 core and
   14 acceptance passes. Retain final debug/release results and separately record
   any 32-bit execution; do not infer either profile or 32-bit coverage from counts.
2. **Documented in PIXEL_API.md:** numeric limits and their scope: per-axis limit,
   construction versus encode acceptance, 4096 versus 256 extras, total stream
   versus scratch memory, per-image versus total-sequence work, and error classes.
3. **Documented in PIXEL_API.md:** global animation override versus plan's strict
   metadata equality,
   exact metadata matching including names, UTF-8 byte length, timing endpoints,
   allowed signed-origin range, alpha blend index limit 10 and per-extra rules.
4. **Documented in PIXEL_API.md:** metadata F16 truncation is not pixel quantization;
   unsupported-depth patch gating is distinct from invalid options, and
   a 16-bit scratch declaration remains a caller proof obligation. Sample-storage
   contracts and the U25..U31 extension's independent-verification limit are explicit.
5. Retain the U31 limitation and execute the focused metadata target against both
   trusted tools. Even a pass is limited to the explicitly exposed fields above;
   coalesced pixels alone do not certify raw layer metadata or arbitrary
   high-depth integer recovery.
6. Run and retain a bounded diverse real-image report; expand only for a concrete
   uncovered route or profile. This harness's stripped, eight-image PNG run is
   not wide-profile/CMYK, JPEG, platform, full-scheduler, or performance acceptance.

No production code was changed by this audit. No full acceptance declaration,
new official pass count, or benchmark claim is implied by this handoff.
