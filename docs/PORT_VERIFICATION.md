# Rust Port Verification and Profiling

Source baseline: optimized C++ libjxl
`b866b3defec13494bfbaf705db93581cda118e37`.
Earlier CLI timing checkpoints: Windows x86_64, Rust 1.97.1, release CLI,
8 encoder threads. Later checks state their own build/input/timing scopes.
Other operating systems are not yet executed; Rust 1.89 compilation is checked
at the checkpoint below, not claimed as a separate cross-platform runtime test.
The complete pixel-lossless migration remains in progress.

## Cooperative and Push-Stream Checkpoint

2026-10-23, Windows x86_64, frozen source still `b866b3d`. This slice connects
the four requested work blocks, without C++ runtime linkage or new ARM work:

- One caller-owned cooperative arena for scheduler roots and nested algorithms;
  bounded MA channel/row roots, disjoint sample loans and whole-subtree jobs;
  LZ stream roots and source logical seed partitions.
- Packed-i16 FJXL prediction/residual/prefix kernels and fused input/YCoCg,
  with native x86 dispatch and the portable scalar fallback.
- Live prefix-cutoff public probe API and preservation of explicit LZ tools
  through source candidate/atlas projection, without changing wire policy.
- Push region-input `FrameEncoder`, area output or explicit current-frame
  fallback, references/crops/zero-duration layers, immutable finalized prefixes,
  bounded optional indexes and metadata boxes.

The final integrated checkpoint passed **1,527 ordinary tests in both debug and
optimized validation**, zero failures: 1,420 core, 44 CLI and 63 integration/API/
doctests. External oracle tests are explicitly enabled and counted separately,
never treated as passes merely because the ordinary run ignored them. Strict
all-target Clippy, formatting, Rust 1.89 all-target compilation and current Rust
x86_64 Linux all-target compilation passed again after the final test additions.
Linux runtime, macOS runtime and ARM are not part of this evidence. Optimized
validation uses LTO off/16 codegen units; the shipped CLI below uses the normal
thin-LTO/one-codegen-unit profile.

Focused verification passed all 126 MA tests, 60 FJXL tests, 16 execution tests,
six parallel small-probe tests and the explicit LZ projection regression.
The external FJXL suites were explicitly enabled: **108 pixel comparisons** and
**32 typed F16-word comparisons** passed official djxl 0.12 and 0.13. Five new
parallel oracle suites passed **28 comparisons**: channel/row roots (4 pixels),
whole-subtree jobs (4 pixels), indexed LZ streams (6 pixels), scheduler batches
(6 pixels) and JPEG reconstruction (8 exact original-byte comparisons).
Profile assertions prove the intended routes ran.

All four new push-sequence suites passed **20 official comparisons** from eight
encodes with one/four threads: mixed full/crop/reference/layers (4), full and
subsampled extras (4), indexed full output and independently decoded suffix
(8), and F16 signed-zero/infinity/NaN-payload words (4). Tests assert actual
area versus explicit buffered routing, release every input lease, and protect
the immutable prefix during every push. Sixteen focused stream unit tests
include late TOC/box-length seek failures, final-index write failures, index
varint boundaries and custom float formats with mixed typed extras. A failed
final index can leave a newly finalized frame prefix; the encoder remains
poisoned and cannot report a successful complete stream. These tests do not
target identical encoded JXL.

The complete main `reference` matrix also passed all **31 explicitly enabled
official-oracle suites** after the shared execution and geometry changes.
The existing `finite_region_input_preserves_original_words` and
`global_layout_frame_area_streaming_float_word_wire` suites were explicitly
rerun afterward and passed both official decoders, covering the older F16/F32
region entry points as well as the new sequence interface.

Independent executor review caught cross-pool entry potentially yielding a root
to another waiting arena loop. The wrapper now rejects that entry before yield;
tests cover retained same-pool/null behavior and reuse after failure. Borrowed
work drains before panic propagation, including secondary payloads whose Drop
could itself panic. Static review found no remaining concrete blocker under the
documented callback contract; this is not a formal concurrency proof.

The normal thin-LTO, one-codegen-unit release CLI was rebuilt. Eight original
PNG cases passed exact decoded-sample comparison through both official decoders.
These are single-run CLI measurements (startup/input decoding included), not
an A/B speedup claim. Input metadata was removed in temporary snapshots without
altering image samples; metadata preservation is tested separately.

| Case | Dimensions | Source PNG bytes | JXL bytes | bpp | CLI seconds | MPix/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Kodak photo | 768x512 | 736501 | 454216 | 9.241048 | 6.516 | 0.0603 |
| Icon | 64x64 | 5087 | 4082 | 7.972656 | 0.140 | 0.0293 |
| Clover texture | 1024x1024 | 2582326 | 1586564 | 12.104523 | 8.782 | 0.1194 |
| Game UI | 634x406 | 175090 | 83134 | 2.583767 | 2.969 | 0.0867 |
| 16-bit photo | 64x64 | 35582 | 17788 | 34.742188 | 0.141 | 0.0290 |
| Gray photo | 500x500 | 120209 | 90224 | 2.887168 | 1.235 | 0.2024 |
| RGBA photo | 500x500 | 374287 | 227480 | 7.279360 | 6.547 | 0.0382 |
| Gray-alpha flower | 510x532 | 176282 | 99031 | 2.919976 | 6.766 | 0.0401 |

All eight used full source-search e9, eight threads, with default optional-tool
switches. Explicit new cooperative tools are covered by the activation oracles,
not assumed enabled by this command. No concurrent build or parent encoding test
ran during these CLI measurements.

Real `wallhaven-vpyekp.jpg` (1920x1280, 140145 bytes) passed another **12 exact
original-JPEG-byte comparisons** with source-search e9, one/four threads,
file/stdin/stdout and failure/output protection. Output was 92752 bytes,
66.183% of input, 0.301927 bpp. CLI times were 0.4520s / 0.3806s and throughput
5.438 / 6.456 MPix/s. This is byte archival, not JPEG-to-pixel coding.

The four work blocks do not imply a Rust decoder, Python ABI, whole-encoder
performance parity or exhaustive input/platform acceptance.

## Previous x86_64 Checkpoint

2026-10-23, AVX2 input/prediction, logical LZ partitions, channel-tree model
decisions and optional JPEG cooperative learning:

- **1,450 workspace tests passed in debug and optimized validation**, zero
  failed: 1,359 core, 44 CLI and 47 other integration/API/doctests. The new
  channel-tree oracle then added two ordinary helper tests, passing in both
  profiles, for **1,452 ordinary tests** at this checkpoint. Ignored external
  tests are recorded separately below.
- Strict all-target Clippy and Rust 1.89 all-target compilation passed. Clippy
  identified one modulo expression in a new test; replacing it with
  `is_multiple_of` changed no encoder code or fixture values. Formatting passed.
  Current Rust's x86_64 Linux all-target compilation also passed after these
  changes; Linux runtime execution and Rust 1.89's missing Linux target remain
  unverified. No ARM target was built.
- Both official djxl 0.12 and 0.13 passed **108 FJXL pixel comparisons** after
  the eight AVX2 input loaders and fused 15/16-bit prediction/prefix path were
  connected. The typed FJXL route passed **32 F16 word comparisons**, including
  signed zero, infinities and NaN payloads. Guarded input/output-page unit tests
  additionally checked the actual native loads and stores.
- Three affected LZ suites passed **172 pixel comparisons**. Logical block and
  seed partition counts now retain configured task/scratch policy even when
  only two physical workers execute eight logical parts. Unit tests check the
  actual cuts and independently reconstruct commands, not just output hashes.
- The new channel/row-tree oracle passed **16 pixel comparisons** over eight
  encodes, including invisible RGB. Runtime profiles proved actual 4/8/16
  partition jobs and 196608 distinct stored records for modes 1/2/3 with
  one/four threads; narrow input had 57344 records and four jobs. Merely setting
  an option was not counted as coverage. Metadata and final-only/probe gates
  have separate unit tests.
- Optional JPEG cooperative MA passed **12 exact original-JPEG-byte checks**:
  4:4:4/4:2:0 restart fixtures, serial execution and one/two bounded cooperative
  roots, through both official decoders. Archived JPEG entropy was erased before
  coefficient encoding. This is not JPEG-to-pixels acceptance.
- No ARM implementation or ARM-specific checks were added. Original C++ source
  was not edited; its existing dirty third_party/zlib state was left alone.

No whole-encoder speedup or complete migration is claimed by these tests.
Channel partition roots are still sequential; inner property/child tools can
use the existing pool. Packed narrow FJXL prediction and exact C++ arena
scheduling remain separate source-execution gaps. The complete port remains
in progress; the new options and their defaults are documented in PIXEL_API.md
and JPEG_API.md.

The normal thin-LTO, one-codegen-unit release CLI was rebuilt after this slice.
Two real PNGs passed file/stdin pixel comparison with both official decoders
at source effort 1 and threads 1/4 (12 comparisons). A real JPEG passed 12
original-byte comparisons at effort 9, including stdin/stdout and failure/output
protection checks. Temporary outputs were removed. Single cold CLI timings,
including process startup, input decoding/reading and output collection:

| Input | Mode / threads | Original bytes | JXL bytes | bpp | Seconds | MPix/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| plant_20.png, 568x746 | Pixel source e1 / 1 | 450529 | 331238 | 6.253785 | 0.1592 | 2.662 |
| plant_20.png, 568x746 | Pixel source e1 / 4 | 450529 | 331238 | 6.253785 | 0.0901 | 4.704 |
| 358colors.png, 72x72 | Pixel source e1 / 1 | 1407 | 1765 | 2.723765 | 0.0301 | 0.172 |
| 358colors.png, 72x72 | Pixel source e1 / 4 | 1407 | 1765 | 2.723765 | 0.0298 | 0.174 |
| wallhaven-vpyekp.jpg, 1920x1280 | JPEG e9 / 1 | 140145 | 92752 | 0.301927 | 0.2766 | 8.884 |
| wallhaven-vpyekp.jpg, 1920x1280 | JPEG e9 / 4 | 140145 | 92752 | 0.301927 | 0.2582 | 9.517 |

These are CLI smoke measurements, not before/after or C++ parity benchmarks.
The tiny palette PNG becomes larger under the explicitly fastest pixel policy;
this does not measure source search or establish a never-larger-than-PNG claim.

## Previous Region Checkpoint

Final follow-up with real region input and seekable output connected:

- **1,416 workspace tests passed in both debug and optimized validation**,
  zero failed: 1,325 core, 44 CLI and 47
  integration/API/doctests. External decoder tests stay explicit, not counted
  as passes merely because Cargo ignored them. Strict all-target Clippy,
  Rust 1.89 all-target compilation and the whole-workspace format check passed.
- Current Rust's x86_64 Linux all-target compile check also passed. Rust 1.89's
  Linux standard-library target is not installed, so that separate combination
  was not verified. No Linux runtime execution is claimed.
- Eleven affected `source_` suites passed in the optimized validation profile:
  **410** official pixel comparisons, including vendored SA-IS, all three LZ
  route suites, MA quantile-64, scoring, actual statistical race and gather.
- FJXL standalone plus region/incremental/seekable routes passed **108** exact
  comparisons. Typed FJXL added **32** exact F16 word checks: sixteen finite
  and sixteen signed-zero/infinity/NaN-payload checks.
- General Modular region input and standard mode-1 TOC backpatch passed **12**
  complete-file decodes, checking both RGBA samples and extra-channel samples;
  depths 8/16 and extra shifts 0/1/3. A separate **16** official word checks
  covered F16/F32 GrayAlpha/RGBA and horizontal/vertical DC-area boundaries.
- The new lower-level region/streaming functions allocate current-area working
  data, not dummy/full input images or a complete compressed-frame buffer.
  Non-Send provider/sink callbacks, exact total limits, reference geometry,
  invalid source eligibility, lease cleanup and I/O poisoning have unit checks.
- The first SA-IS test caught upstream's singleton parallel-iterator use. Its
  local pure-Rust serial branches now pass isolated global-pool initialization
  tests; no native C/C++ build or workaround global pool was introduced.
- Two failed tests were corrected without weakening pixel correctness: the
  new FJXL header-reuse fixture needed to reset its consumed output cursor each
  iteration; an old acceptance assertion still rejected the now-legal 262145
  axis. The official pixel checks independently confirm the latter boundary.

The optimized validation profile disables LTO and uses 16 codegen units only
for test builds. Normal release defaults remain thin LTO and one codegen unit.
General region output currently exposes one full-canvas frame; it is separate
from the existing multiframe API and does not imply a runtime decoder or Python
ABI. Remaining x86 optimized kernels and cooperative execution differences
still need source-port review. No new ARM development or ARM checks were run.

2026-10-23, after prepared MA storage, optional MA scoring/proposals and FJXL
entropy SIMD integration. No further ARM work is included.

- The debug workspace passed **1,376 tests**: 1,285 core, 44 CLI and 47
  integration/API/doctests. Strict all-target Clippy and Rust 1.89 all-target
  compilation passed. The subsequent long-axis geometry correction and ongoing
  SA-IS/region-callback work require separate validation.
- FJXL's 76 standalone official comparisons were rerun after its entropy-SIMD
  changes and passed. New generic/chain/range LZ routes passed **68** comparisons.
  Quantile-32 and refined proposals passed **32** comparisons; the subsequently
  added quantile-64 fixture is not in that count. The separate 1024x512 Gray16
  statistical-race case passed both official decoders: 524287 stored records,
  `race_nodes=1`, `race_draws=3584`, `race_pruned=0`. This proves the race ran,
  not that it pruned or accelerated this deliberately noisy fixture.
- Level 10 axes through 2^30 and total pixels through 2^40 now follow the
  source limits, subject to explicit caller resource limits. Both official
  decoders passed **four** checks for 262145x1 and 1x262145 images. No test
  allocates or claims to encode a 2^40-pixel image.
- The normal thin-LTO release CLI passed all eight diverse real pixel cases
  below through official djxl 0.12 and 0.13. The harness removes conflicting
  metadata and compares original decoded sample values, including invisible
  RGB. This is pixel acceptance, not metadata-profile acceptance.
- The same CLI passed **36** original-JPEG-byte checks with `--source-search`,
  efforts 3/9 and threads 1/4 over `t2/145702319_p0.jpg`,
  `wallhaven-3q3z9d.jpg` and `wallhaven-k81776.jpg`. File/stdin/stdout, refusal
  to overwrite, same-path rejection and invalid-input output protection passed.

Single-run measurements: Windows x86_64, eight encoder threads, source-search
effort 9, normal release profile. CLI wall time includes input decoding and
process I/O, excludes official decoding, and is not a warmed benchmark or
C++ performance-parity claim. Other agents were editing sources; no other
parent build/test command overlapped this corpus run.

| Case | Dimensions | Source PNG bytes | JXL bytes | bpp | CLI seconds | MPix/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Kodak photo | 768x512 | 736501 | 454216 | 9.241048 | 6.922 | 0.0568 |
| Icon | 64x64 | 5087 | 4082 | 7.972656 | 0.141 | 0.0290 |
| Clover texture | 1024x1024 | 2582326 | 1586564 | 12.104523 | 9.531 | 0.1100 |
| Game UI | 634x406 | 175090 | 83134 | 2.583767 | 3.203 | 0.0804 |
| 16-bit photo | 64x64 | 35582 | 17788 | 34.742188 | 0.172 | 0.0238 |
| Gray photo | 500x500 | 120209 | 90224 | 2.887168 | 1.250 | 0.2000 |
| RGBA photo | 500x500 | 374287 | 227480 | 7.279360 | 6.860 | 0.0364 |
| Gray-alpha flower | 510x532 | 176282 | 99031 | 2.919976 | 7.312 | 0.0371 |

This does not complete migration/performance acceptance. The newly connected
region input/within-frame streaming and remaining source execution differences
are tracked in the feature-specific status documents.

## Frame Search and JPEG Deduplication Checkpoint

Subsequent Windows x86_64 debug integrations, independently executed:

- FJXL standalone scalar/native profiles: **76** official comparisons, covering
  19 cases and both official decoders. Narrow depths use exact integer recovery
  from NPY; the initial PNG comparison failed because official UINT8 rendering
  applies dither. See FJXL_PORT_STATUS.md, including the reproduced failing value.
- Typed FJXL dispatch: **32** exact U8/U16 pixel checks plus **16** exact F16
  word checks, including signed zero, subnormals and negative values.
- Ordered streaming layers/animation: **12** official comparisons. Indexed
  full-animation and independently extracted suffixes: **8** decode checks,
  with parsed offsets, durations and timebase checked separately.
- New source block-LZ variants: **36** exact pixel comparisons. Optional MA
  histogram tools: **16**; actual sparse/stripe gather: **12**; new count-log,
  sliding clipped-entropy and FastEntropy/description profiles: **48**.
- Intrinsic display dimensions: **20** official metadata-parser checks, without
  resampling the encoded pixels. Existing ARM kernels passed an earlier compile
  check only; further ARM-specific development/testing stopped at user request.

These records precede the later SA-IS and additional FJXL entropy-SIMD changes;
they do not implicitly certify subsequent edits or performance.

2026-10-23, Windows x86_64, debug validation (not a speed benchmark):

- The first integrated checkpoint passed **1,285** workspace tests: 1,194
  core, 44 CLI and 47 integration/API/doctests. A following incremental-container
  checkpoint passed **1,302**: 1,211 core, the same 44 CLI and 47 others.
  Concurrent MA prepared-column integration was not wired at that snapshot;
  dead-code warnings remain until that handoff is complete.
- JPEG metadata deduplication passed **30** original-byte checks through
  official 0.12 and 0.13. ICC/Exif/XMP match actual emitted sources; duplicate
  and mismatched APP bytes stay opaque. These are new runs, not old results
  relabeled for the changed serializer.
- The new coefficient-only JPEG source search passed **176** original-byte
  checks: four fixtures, effort 1..11, threads 1/4, two official decoders.
  The test checks the exact candidate prefix and actual minimum output with
  fixed reconstruction quality. It does not compare new JXL bytes to old JXL.
- The three-configuration synthetic JPEG regression passed **114** checks
  (19 fixtures, two decoders). This is not the earlier 12-configuration/456
  matrix. Its 8.99s suite time includes fixture generation and official decoding.
- Existing and newly searched layers/animation each passed **12** official
  pixel checks: 8/16-bit RGBA, animation, true zero-duration layers and explicit
  saved reference frames. Winners are real context-bearing frame encodings,
  not concatenated single-image files.
- Lightning inputs reaching Modular now normalize to Thunder unconditionally,
  following frozen enc_frame.cc:2924. FJXL remains separate port work.

The newer code under active integration includes MA compact-column handoff,
optional histogram tools, further LZ tools, intrinsic dimensions and native
AArch64 kernels. Later build/oracle results must be recorded before claiming
those changes verified. Current CLI binaries may precede these source edits.

## Shared JPEG and Pixel Tools Checkpoint

2026-10-23, Windows x86_64. The coefficient composer now uses source DC,
AC-metadata and quant-table learning domains, merged by real stream ID, with
shared auxiliary entropy. Its public byte/writer API is separate from pixel
encoding. Exif orientation/boxes and standard XMP boxes are now composed while
all original APP bytes remain in `jbrd`.

- **696 official original-JPEG-byte checks passed** for the shared model:
  456 synthetic combinations, 102 checks over all 17 real JPEGs, 44 all-tier
  checks, 16 execution-setting checks, 30 metadata checks and 48 checks through
  the actual byte/writer adapter. No complete-frame test invokes an encoder.
- The new real-corpus run first rejected the final 6144x3321 slow-tier case at
  the one-million aggregate MA sample limit, after 100 exact checks. The JPEG
  default is now explicitly 4,194,304 selected samples; the successful rerun
  did not reduce sampling, lower effort, or fall back to pixels.
- Real/tier/metadata/adapter checks used a validation-only optimized profile:
  LTO disabled and 16 codegen units. Twenty focused tests including the
  three-configuration synthetic matrix completed in 95.13s. The separate full
  12-configuration synthetic rerun took 26.80s. These are test-suite times,
  not warmed CLI compression benchmarks or C++ speed comparisons.
- Pixel tool additions include default-off MA predictor/property equivalence
  reuse, LZ77 packed values/index coordinates and compact DP node storage.
  All four affected `source_` reference suites passed: 80 tier, 68 LZ route,
  16 MA dedup and 16 parallel-gather pixel comparisons through both official
  decoders, **180** total. These compare decoded pixels, not JXL bytes.
- The normal release CLI build passed. `tools/check_jpeg_cli.py` passed **44**
  original-byte comparisons: four real images, effort 3/9, threads 1/4, binary
  stdin/stdout and file output. Existing-output refusal, explicit replacement,
  same-path refusal and invalid-input output protection were also checked.
- The initial full debug workspace checkpoint passed 1,137 core tests,
  43 CLI tests and 47 API/integration/doctests, **1,227** total. Strict Clippy
  passed after removing two redundant test allocations; formatting passed.
  Later resource-error classification edits require the follow-up below.

Final follow-up after the resource-error fix:

- Full debug and optimized-validation workspaces each passed **1,235 tests**:
  1,145 core, 43 CLI and 47 API/integration/doctests; no failures. External
  oracle tests remain opt-in and are counted separately, not as implicit passes.
  The optimized validation build used LTO=false and codegen-units=16; manifest
  release defaults were not changed. That build took 2m14s, not encode time.
- All 180 affected source/pixel comparisons also passed in that optimized
  build; the four suites took 11.06s. The public JPEG byte/writer suite was
  rerun afterward and all 48 official byte comparisons passed in 2.81s.
- Final strict Clippy, formatting and Rust 1.89 all-target checks passed.
  Cross-target all-target checks passed for x86_64 Linux, aarch64 Linux and
  aarch64 macOS. These are compile checks, not OS/ARM runtime tests.
- Budget/output/allocation failures retain `Error::Limit`; malformed and
  unsupported JPEG inputs remain `Error::InvalidInput`. Eight focused tests
  cover nested metadata/DC/AC failures and untouched sinks.
- The normal thin-LTO CLI was rebuilt after that fix (1m23s build time).
  A final e3/e9, four-thread check on `t2/145702319_p0.jpg` passed all 12
  original-byte comparisons and file/pipe/failure checks. No additional source
  edits followed; no commit or push was made, and the frozen C++ tree was
  unchanged (its pre-existing dirty zlib submodule was left alone).

CLI measurement samples from that single acceptance run (normal thin-LTO
release, 4 threads, wall time includes process startup/read/encode/stdout,
not official decoding). These are not warmed benchmarks or C++ comparisons;
the first part of the run overlapped a Clippy process.

| JPEG | Dimensions | Effort | Original bytes | JXL bytes | bpp | Seconds | MPix/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wallhaven-3q3z9d.jpg | 3840x1635 | 3 | 2237363 | 1876158 | 2.390619 | 0.3804 | 16.506 |
| wallhaven-3q3z9d.jpg | 3840x1635 | 9 | 2237363 | 1871533 | 2.384726 | 3.4860 | 1.801 |
| wallhaven-jexkwm.jpg | 3840x2160 | 3 | 1899821 | 1543147 | 1.488375 | 0.3036 | 27.318 |
| wallhaven-jexkwm.jpg | 3840x2160 | 9 | 1899821 | 1538882 | 1.484261 | 4.2843 | 1.936 |
| t2/145702319_p0.jpg | 2000x2183 | 3 | 1142283 | 940933 | 1.724110 | 0.1951 | 22.381 |
| t2/145702319_p0.jpg | 2000x2183 | 9 | 1142283 | 934721 | 1.712727 | 1.2436 | 3.511 |
| wallhaven-yqxzqx.jpg | 5120x2880 | 3 | 9472248 | 8066463 | 4.376336 | 1.1290 | 13.061 |
| wallhaven-yqxzqx.jpg | 5120x2880 | 9 | 9472248 | 8044994 | 4.364689 | 9.7587 | 1.511 |

The full pixel-lossless source-tool port, optimized JPEG search, metadata
deduplication, runtime Rust decoder and Python/C ABI remain open. No source
hash/whole-JXL identity requirement has been introduced.

## Earlier JPEG Frame and MA Storage Checkpoint

2026-10-23, Windows x86_64:

- Current debug workspace: 1,078 core tests passed (five separately enabled or
  optional tests ignored), 10 API tests, 14 pixel/resource acceptance tests,
  35 CLI tests, ordinary integration helpers and the doctest passed.
- Strict all-target Clippy, formatting check and
  `cargo +1.89.0 check --locked --workspace --all-targets` passed. A final fixture
  checked-division-only edit was retested independently after the workspace run.
- The preceding normal-release core snapshot passed 1,051 tests. That snapshot
  includes source MA u16 saturated counts/two-probe deduplication and the first
  JPEG auxiliary tools, but predates the final JPEG stream-policy/ICC integration.
  Do not present it as current full release coverage.
- Eighty source-tier pixel comparisons passed both official djxl versions after
  shared header/inline Modular writer and MA-storage changes. The earlier 47
  release oracle suites are historical and were not all rerun at this checkpoint.
- Complete, private Rust JPEG coefficient frames passed 456 synthetic checks
  (19 fixtures x 12 configurations x two decoders) and 102 real-image checks
  (17 images x three configurations x two decoders). All original JPEG bytes
  matched after archived entropy was cleared before encoding. Neither complete
  frame test invokes cjxl. See JPEG_TOOL_PORT_STATUS.md for detailed coverage,
  reproduction commands and the validation-only release-profile overrides.
- Separate Rust jbrd/container tests passed 136 checks with official coefficient
  frames as fixtures. Those are not included in the 558 full-Rust-frame checks.

At that earlier checkpoint the coefficient composer remained cfg(test)-only. No public archival API, CLI
transcode command, runtime Rust decoder or completed migration is claimed.
There are no new commits/pushes, and the frozen C++ source was not edited.

## Measurement Contract

- Correctness compares decoded samples, including hidden RGB, not JXL hashes.
- `tools/check_png_cli.py` checks 8-bit PNG pixels using independent official
  djxl 0.12 and 0.13. It deletes temporary decoder outputs afterward.
- CLI time includes process startup, input decode, encoding and stdout capture;
  it excludes oracle decoding. Core-only timing is a different metric.
- `RS_JXL_PROFILE=1` reports cumulative inclusive stage times on stderr.
  `RS_JXL_PROFILE=trace` also reports begin/end markers, useful on timeout.
  Nested and parallel stage times overlap: do not add them as wall time.
- Profiling is off by default and does not change sampling or encode settings.
  Diagnostic I/O and timers can affect timing. Use unprofiled runs for speed
  comparisons; no performance claim is inferred from test counts.
- `--timeout` is a per-process test guard, not an encoder fallback. Timeouts
  are failures without an output size or successful correctness check.

Example from the Rust repository (replace decoder paths as needed):

```powershell
python -B tools/check_png_cli.py --encoder target/release/rs-jxl.exe `
  --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe `
  --decoder ../jxl_python/out/encoder_opt_baseline/djxl.exe `
  --threads 8 --skip-stdin-check --profile summary `
  --encoder-arg=--source-effort --encoder-arg=9 `
  --encoder-arg=--source-sample-limit --encoder-arg=33554432 IMAGE.png
```

## Before Default Routing Correction

These measurements describe the incorrect always-shared-tree projection, not
equivalence with C++ e9. Single candidate, not the full multi-candidate search.
The explicit sample ceiling is 33,554,432; effort and sample fraction are not
lowered to meet this ceiling.

| Input | Dimensions | PNG bytes | JXL bytes | bpp | CLI seconds | MPix/s | Correctness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `9bde6aaf390109f97622d35ff1e900ee.png` | 1264x1152 RGBA | 323776 | 275120 | 1.511515 | 17.1263 | 0.085 | Both djxl versions exact, file and stdin |
| `test_img/wallhaven-mlzoy1.png` | 3840x2160 RGB | 14570359 | not available | not available | timeout at 120 | not available | Not verified |

Natural-image process working set was observed at 5,544,943,616 bytes during
execution. This is one observation, not a peak-memory measurement.

A subsequent instrumented screenshot run took 20.6530 CLI seconds and still
decoded exactly. Selected inclusive profile times:

| Stage | Seconds | Calls |
| --- | ---: | ---: |
| MA learning | 12.877479 | 2 |
| MA sample preparation (inside learning) | 4.449189 | 2 |
| MA node training (inside learning) | 3.872946 | 2 |
| MA sample gathering (inside learning) | 2.416197 | 2 |
| Shared LZ77 parsing | 4.989845 | 2 |
| Entropy model building | 1.660611 | 15 |
| Patch analysis | 0.199989 | 1 |

Main-frame learning gathered 2,814,598 records and retained 2,814,465 distinct
records. The constant-suffix branch additionally deep-copied variable samples.
The source default frame-area routing was absent from native projection:
`CanDoStreamingEncoding` permits e9 pixel lossless with more than eight groups
unless another gate, such as explicitly forced patches, disables streaming.
The Rust helper algorithms existed, but native projection did not select them.

## Checkpoint Status

After opt-in profiling was added, `cargo test --locked --workspace` passed:
508 core tests, 34 CLI tests, 10 encoding API tests, non-oracle helper tests
and the doctest. Ignored official-oracle tests require separate execution;
this ordinary test run does not claim those tests passed again.

## After Default Routing Correction

Unprofiled, same input paths, e9 single candidate, eight threads and the same
33,554,432 sample ceiling. All outputs passed both official decoder versions.
This change restores the default execution domain; it is not a claim that the
default local path minimizes bytes for screenshots.

| Input | JXL bytes | bpp | CLI seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| `9bde6aaf390109f97622d35ff1e900ee.png` | 411621 | 2.261455 | 2.8062 | 0.519 |
| `test_img/wallhaven-mlzoy1.png` | 14178878 | 13.675615 | 19.2583 | 0.431 |

The screenshot becomes faster but larger than the previous whole-frame route.
The natural image now finishes within the 120-second test guard. Its process
working set was observed at 451,416,064 bytes, with the Windows process peak
counter at 612,667,392 bytes at that observation; this is not a continuous
whole-run memory trace.

For context, the existing official 0.12 executable (`a1ebdf2`, Clang 17.0.6)
was run separately with `-d 0 -e 9 --num_threads=8 --quiet` and verified by
pixel comparison. This is not the frozen optimized fork, and C++ input metadata
handling is not normalized to the Rust `--strip-metadata` run. Its CLI timing
includes writing a temporary file, whereas Rust uses captured stdout.

| Input | Official 0.12 JXL bytes | bpp | CLI seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Screenshot | 432908 | 2.378406 | 0.9302 | 1.565 |
| Natural image | 14183090 | 13.679678 | 6.2540 | 1.326 |

Compression is close in these cases, but Rust remains roughly three times
slower. Neither source-policy nor performance parity is certified by this run.

After the routing/sample/entropy fixes, the ordinary debug workspace tests
passed (520 core, 34 CLI, 10 encoding API, helpers and doctest), and strict
all-target Clippy passed. The Python PNG helper's three tests also passed;
16-bit PNG is explicitly rejected rather than silently narrowed by Pillow.

The corrected natural-image profile (`target/profile-natural.log`) completed
with exact pixels and 20.9806 CLI seconds. Cumulative inclusive work times were
125.210996 seconds for 135 MA learners, 79.483292 for node training,
58.540759 for split sweeps, 25.390902 for sample preparation and 16.781187 for
sample gathering. These parallel/nested totals are not wall-clock times.
Local LZ77 accumulated 8.293757 seconds; entropy model building 1.340742.
The remaining primary hotspot is MA learning, not entropy output serialization.

After aligning the local learner's source group ID (0 for learning, actual
wire stream ID for tokenization), all independently decoded suites passed:
14 main, 16 float, two Add-patch and two metadata/frame suites. The native
streaming boundary added 24 exact pixel comparisons with the two decoders.
The release workspace suite also passed: 520 core, 34 CLI, 10 API, helper
tests and doctest. Strict all-target Clippy passed again after that ID fix.

The default MA arithmetic mismatch at that checkpoint is recorded in
[MA_SCORING_ALIGNMENT.md](MA_SCORING_ALIGNMENT.md). It was corrected in the
subsequent checkpoint below; retain the earlier numbers as historical evidence.

## Complete Native Scheduler Check

The current release CLI was rebuilt after the local learner-ID fix. On the
same 1264x1152 screenshot, adding `--source-search` to the e9 settings produced:

| Full encodes | Probe encodes | Winner effort | JXL bytes | bpp | CLI seconds | MPix/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 134 | 9 | 146660 | 0.805753 | 44.1160 | 0.033 |

Both official decoders exactly reproduced all pixels. This is substantially
smaller than the default single-candidate output, but searching is still slow.
It verifies that the native driver reaches real encodes and retains its output;
it does not certify all optional C++ search paths or their performance. The
official 0.12 single-candidate numbers above are not a full-search counterpart.

No C++/Python project source was modified for these checks. No neural model,
third-party JXL encoder or linked C++ codec is used by the Rust implementation.

## Default MA Scoring Correction

The trainer now uses source-default f32 FastLog2f with fused histogram moves,
native SSE2/AVX2, weighted per-bin totals and the small-population table.
It no longer silently uses the optional f64 incremental entropy path. Source
sampling fraction, properties/predictors and complete-search gates are unchanged.

Same-session comparison of the retained pre-change CLI and corrected CLI,
eight threads, e9 single candidate, metadata stripped, sample ceiling 33554432:

| Image | Before bytes | After bytes | Before seconds | After seconds | After bpp | After MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Screenshot | 412813 | 412541 | 3.2758 | 2.7954 | 2.266510 | 0.521 |
| Natural image | 14179361 | 14179167 | 16.8496 | 12.3221 | 13.675894 | 0.673 |

Single-run timings include CLI startup/input decode/stdout capture; no concurrent
build or test workload ran during measurement. Both official decoders reproduced
all samples. These numbers do not establish C++ performance parity or population
statistics. The old binary includes the local learner-ID fix, so its output is
not identical to the older route-fix checkpoint above.

Central debug and release workspace tests passed: 578 core, 10 API, 34 CLI, helpers and
doctest. Strict all-target Clippy passed. After the correction, all ignored
official suites were explicitly rerun with BOTH djxl versions: 16 float,
two metadata/frame, two Add-patch, and 14 main suites passed. No compressed
JXL byte identity was used as the acceptance criterion.

JPEG prerequisite work is separately recorded in
[JPEG_TOOL_PORT_STATUS.md](JPEG_TOOL_PORT_STATUS.md). The real 17-file archive
test preserved 33,132,526 input bytes, including three progressive JPEGs. It is
not an independent coefficient or JPEG-to-JXL reconstruction test.

The full native scheduler was also measured with `--source-search` on the
screenshot using both retained binaries in the same session:

| Version | JXL bytes | bpp | CLI seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Before default MA correction | 146660 | 0.805753 | 34.2735 | 0.042 |
| After default MA correction | 146660 | 0.805753 | 27.5230 | 0.053 |

Both outputs independently decoded to the exact original samples with both
official decoders. Equal byte counts here are an observation, not an encoding
target or an assertion that the compressed byte sequences must match. The
same-session baseline is faster than the older 44.1160-second run above;
do not attribute that unrelated timing variation to this change.

## Probe Batching And Null Runner

The next checkpoint wires legacy small-probe control/screen/shortlist/validation
and confirmation into ordered batches. Independent encodes run serially inside
the outer caller pool; they do not enter another pool or dispatch nested Rayon
tasks. Public `threads=1` encoding now also avoids allocating a worker thread.
The source responsive candidate is separate from local squeeze directions.

This performance comparison is **Rust before versus Rust after**, NOT Rust
versus the C++ encoder. Official djxl executables are independent pixel oracles
only. It cannot establish compression or execution parity with the frozen fork.
Same-session, one run per version, 1264x1152 PDF screenshot, eight threads,
`--source-effort 9 --source-search --source-sample-limit 33554432`, stripped
metadata, CLI startup/input decoding/stdout capture included:

| Rust version | JXL bytes | bpp | CLI seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Before probe batching | 146660 | 0.805753 | 35.4759 | 0.041 |
| After probe batching/null runner | 146660 | 0.805753 | 24.3892 | 0.060 |

Both official 0.12/0.13 decoders reproduced all original samples, including
hidden RGB. No concurrent build/test workload ran during measurement. The
observed 31% wall-time reduction is a single-pair observation, not a general
speedup claim. Identical output sizes are not a requirement for future changes.
No encoder hash comparison was used. Other machines/platforms remain untested.

Central checks after integration passed: 594 core, 10 API and 34 CLI tests,
helper tests and doctest in both debug and release; strict all-target Clippy
and formatting check. All ignored oracle suites were explicitly run with
BOTH official decoders: 15 main, 16 float, two metadata/frame and two Add-patch
suites passed. The new native responsive matrix accounts for 120 independent
pixel comparisons across 8/16-bit Gray/RGB/RGBA, threads 1/4 and source tiers
0/1/4/7/9. Single-thread cases now exercise the core null-runner path. The JPEG
archive ignored test was not rerun because this checkpoint changes no JPEG tools.

Complete pixel migration remains open: optional source scheduling routes,
explicit effort 11, unsupported patch/subsampled-extra combinations and
cross-platform execution evidence remain. JPEG-to-JXL archival transcoding
has not been started as the next acceptance phase.
## Frozen-Source Comparison, 2026-10-23

The isolated C++ executable was built from the frozen source during this run;
see [build provenance and version caveat](CPP_REFERENCE_BUILD.md). Source was
not edited. The generic harness records externally supplied binaries as
unverified rather than inferring provenance from their version strings.

One cold sample per configuration, eight threads, effort 9, sample limit
33,554,432. These are exploratory CLI wall times, including startup and PNG/JXL
I/O but excluding verification. No competing build/benchmark ran concurrently.
Both encoders received the same metadata-free PNG, and BOTH official djxl
0.12 and 0.13 independently recovered all original samples, including hidden RGB.

| Image / mode | C++ bytes | Rust bytes | C++ bpp | Rust bpp | C++ seconds | Rust seconds | C++ MPix/s | Rust MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PDF 9bde, single candidate | 412714 | 412539 | 2.267460 | 2.266499 | 1.071 | 3.021 | 1.359 | 0.482 |
| PDF 9bde, full scheduler | 146590 | 146658 | 0.805369 | 0.805742 | 14.269 | 24.582 | 0.102 | 0.059 |
| wallhaven-49yekn, single candidate | 708307 | 707442 | 2.732666 | 2.729329 | 0.820 | 2.350 | 2.528 | 0.882 |
| wallhaven-49yekn, full scheduler | 699050 | 698852 | 2.696952 | 2.696188 | 3.654 | 7.592 | 0.568 | 0.273 |

Dimensions: PDF 1264x1152 RGBA8; wallhaven 1920x1080 RGBA8. Single-candidate
C++ uses explicit `--iterations=50` to retain the default learning fraction
while disabling its outer scheduler. Full-search traces were required and
observed. Matching effort labels are not proof of complete policy equivalence.
The size differences are small here; the Rust latency gap is still material.
This is NOT an integer-versus-float or SIMD speedup measurement.

Raw local reports: `out/port_cpp_comparison_20261023.{json,md}`. Temporary
compressed and decoded outputs were removed by the harness. This measurement
identifies Rust binary SHA256
`d1570e8fb6bd1d6c97b120d0fd2dbd6b8d8e01d5dea7b6c003f0a9b3d9bf69b4`;
later source changes require another measurement.

## Fused Integer Cost And Entropy Writer Checkpoint

Same-session Rust before/after comparison, Windows x86_64, effort 9, eight
threads, sample limit 33,554,432. Each configuration received one warmup per
binary followed by three alternating timed repetitions. Times are medians of
complete CLI invocations, excluding independent decoding checks. There was no
concurrent build or test workload during the measurements.

| Image / mode | Bytes, both versions | bpp | Before seconds | After seconds | After MPix/s | Throughput speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PDF 9bde, single candidate | 412539 | 2.266499 | 2.855 | 2.704 | 0.539 | 1.056x |
| PDF 9bde, full scheduler | 146658 | 0.805742 | 22.586 | 22.306 | 0.065 | 1.013x |
| wallhaven-49yekn, single candidate | 707442 | 2.729329 | 1.943 | 1.935 | 1.071 | 1.004x |
| wallhaven-49yekn, full scheduler | 698852 | 2.696188 | 6.964 | 6.555 | 0.316 | 1.063x |

Every measured output recovered the original samples through both official
0.12/0.13 decoders. Equal sizes are observations, not a compressed-byte identity
requirement. These two images and three repetitions do not establish a general
speedup; in particular, the smallest differences may be measurement variation.

The after binary includes the fused i32 cost kernel AND direct entropy-token
writing/single-context model changes. This is a checkpoint bundle comparison,
not an attribution of all improvement to SIMD. The isolated cost-kernel
benchmark is recorded separately in [INTEGER_AND_FLOAT_PATHS.md](INTEGER_AND_FLOAT_PATHS.md).
MA learning and other search work still dominate some complete encodes.

Harness: `tools/compare_rust_checkpoints.py`. Retained local report:
`out/integer_cost_checkpoints_20261023.json`, including binary identities,
per-trial measurements and decoder results. Temporary encoded/decoded files
were removed. The before binary is `out/rs-jxl-before-integer-cost.exe`.

## Official 0.12 / Optimized C++ / Rust

The following is a new three-way measurement, not a mixture of historical
timings. Same two metadata-stripped PNGs, lossless effort 9, eight threads,
one warmup and three timed repetitions per encoder/image. Serial CLI wall
time includes process startup and input/output, but excludes validation.
Encoder order rotates each repetition; no build or test ran concurrently.
The optimized C++ and Rust use their full outer scheduler, confirmed by their
diagnostics. Official 0.12 uses default e9, without explicit learning overrides.
Equal effort numbers do NOT mean equal search work or compressed size.

| Image | Encoder | Bytes | bpp | Median seconds | MPix/s |
| --- | --- | ---: | ---: | ---: | ---: |
| PDF 9bde | Official 0.12 | 432906 | 2.378395 | 0.909 | 1.601 |
| PDF 9bde | Optimized C++ b866b3d | 146590 | 0.805369 | 13.453 | 0.108 |
| PDF 9bde | Rust current | 146658 | 0.805742 | 22.480 | 0.065 |
| wallhaven-49yekn | Official 0.12 | 708307 | 2.732666 | 0.932 | 2.225 |
| wallhaven-49yekn | Optimized C++ b866b3d | 699050 | 2.696952 | 3.452 | 0.601 |
| wallhaven-49yekn | Rust current | 698852 | 2.696188 | 6.742 | 0.308 |

On these two inputs, optimized C++ saves 66.138% / 1.307% against default
official 0.12, at 14.794x / 3.704x wall time. Rust has very similar output sizes
to optimized C++, but takes 1.671x / 1.953x its time. This is a material port
performance gap, not evidence that the port is finished. These two selected
inputs cannot characterize a general dataset. This comparison also does not
isolate the fork's changes from upstream changes since 0.12.

All 24 encoded outputs, including warmups, passed both official decoders:
48 exact original-sample comparisons including invisible RGB. Encoded-byte
identity is not tested or required. The new harness is
`tools/compare_three_encoders.py`; raw local report:
`out/official_cpp_rust_20261023.json`. Temporary image files were cleaned.

Binary provenance:

- Official 0.12 is the preserved local build documented in the original
  project's `docs/JXL_OFFICIAL_012_COMPARISON.md`, not a downloaded release
  artifact. Its unchanged SHA256 is
  `6b69b96a572fd2a7e6b2baf8968d726998385ee9ad6d4219d171c0a1f1082b88`.
  That audit matched encoder-core/tools sources to upstream v0.12.0; local
  build compatibility and an extras API fix are documented there.
- Optimized C++ is the isolated frozen-source build described in
  [CPP_REFERENCE_BUILD.md](CPP_REFERENCE_BUILD.md). Its version string embeds
  the parent repository revision; it is not the source commit identifier.
- Rust SHA256:
  `b40e58075ac4b4ab20c31ae507d063f0cfd928150deb1b2f52b92cf38a7140d9`.
  These hashes identify executables, not a compressed-output correctness goal.

## Pixel Tool Integration Checkpoint

2026-10-23, Windows x86_64. The 869-core checkpoint passed in debug and release,
with 10 API tests, 14 explicit pixel acceptance/resource tests and 35 CLI tests.
All 56 release official-oracle suites passed with both independent decoders,
including the added source LZ routes, channel-parallel token generation, palette
ordering and metadata-field checks. This predates the subsequent cache-fusion,
trial byte-budget collector and JPEG entropy-writer edits.

`tools/verify_port_corpus.py` additionally checked eight diverse originals with
full e9 source scheduling, eight threads and both decoders. Every decoded sample
matched; outputs were temporary. Inputs were metadata-free PNG snapshots, so
this corpus run does not test metadata retention. Synthetic suites separately
test nonzero invisible RGB; the real alpha images here did not contain it.
The RGB16 case contains 12,170 samples not representable as expanded U8.

| Input category | Shape | Input PNG bytes | JXL bytes | bpp | CLI seconds | MPix/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Kodak photo | 768x512 RGB8 | 736501 | 454216 | 9.241048 | 8.547 | 0.046 |
| Icon | 64x64 RGBA8 | 5087 | 4082 | 7.972656 | 0.172 | 0.024 |
| Clover texture | 1024x1024 RGB8 | 2582326 | 1586564 | 12.104523 | 10.437 | 0.100 |
| Game UI | 634x406 RGBA8 | 175090 | 83134 | 2.583767 | 4.140 | 0.062 |
| Nikon crop | 64x64 RGB16 | 35582 | 17788 | 34.742188 | 0.156 | 0.026 |
| Gray | 500x500 Gray8 | 120209 | 90224 | 2.887168 | 1.500 | 0.167 |
| RGBA | 500x500 RGBA8 | 374287 | 227480 | 7.279360 | 8.750 | 0.029 |
| Gray-alpha | 510x532 GA8 | 176282 | 99031 | 2.919976 | 8.000 | 0.034 |

These are one-pass corpus timings, not repeated performance claims. Full source
paths and bounded runner policy are in PIXEL_ACCEPTANCE_AUDIT.md; raw report:
`out/pixel_port_corpus_20261023.jsonl`, eight successes and no skips/timeouts.

The separate `tools/compare_rust_checkpoints.py` run used one warmup and three
alternating timed repetitions per image/binary, e9 full scheduling, eight threads,
sample ceiling 33,554,432, no concurrent build/test and both official decoders.
All sixteen outputs (including warmups) passed 32 exact sample comparisons.

| Image | Before bytes | After bytes | After bpp | Before median seconds | After median seconds | After MPix/s | Throughput speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PDF 9bde | 146658 | 146658 | 0.805742 | 24.016 | 21.452 | 0.068 | 1.119x |
| wallhaven-49yekn | 698852 | 698853 | 2.696192 | 6.952 | 6.064 | 0.342 | 1.146x |

The one-byte size increase is accepted; JXL identity is not required. This
measures the complete checkpoint bundle, not one isolated tool or SIMD kernel.
The C++ performance gap is not closed, and two cases do not establish a general
speedup. Do not combine these timings with the earlier three-encoder run as
though they were measured together.

Raw report: `out/port_completion_compare_20261023.json`. Preserved executable
`out/rs-jxl-pixel-checkpoint.exe` has SHA256
`667b2ac593b0b080813b6f5c75f7922dc5b4417bb72e018d3dd3e2561d645c3a`.
The before artifact `out/rs-jxl-before-port-completion.exe` has SHA256
`b40e58075ac4b4ab20c31ae507d063f0cfd928150deb1b2f52b92cf38a7140d9`.
Both identify executables only. The preserved after CLI predates later MA
fusion, candidate-bound cancellation and JPEG-writing changes.

## JPEG Coefficient And Reconstruction Checkpoint

The JPEG MCU reader was independently checked against a development-only
libjpeg-turbo coefficient extractor on all 17 original JPEGs: 261,138,944 real
quantized coefficients matched exactly. Inputs included three progressive files
and one restarted file. Libjpeg does not expose transmitted dummy edge blocks;
those are covered by synthetic known-coefficient fixtures, not this oracle.
The extractor is not linked into the Rust product or used for encoding.

The new `jpeg::scan_writer::reconstruct_from_coefficients` was then checked
after clearing and releasing every archived entropy payload. All 17 JPEGs,
totalling 33,132,526 bytes, were rebuilt byte-for-byte from coefficients, table
snapshots, EOB/ZRL hints, padding and marker records. This is stronger than
archival replay, but still does NOT test JPEG-to-JXL coefficient/frame coding.
No reconstructed files were written by this acceptance test.

The 95 JPEG unit tests passed, including explicit bit-pattern oracles, signed
DC/AC refinement, maximal EOB buffering, preserved marker fill, output budgets,
failure-state poisoning and mutated-public-structure rejection. The following
full debug checkpoint passed 938 core tests (two opt-in tests separate), 10 API,
14 pixel acceptance and 35 CLI tests, ordinary integrations and the doctest.
Strict all-target Clippy also passed before subsequent target/column changes.

The separate native candidate-cancellation oracle initially failed because its
tier-8 fixture replaced the requested predictors with a predefined source tree.
It now uses verified tier-6 Top/Left candidates with identical geometry and no
LZ77. The repaired test passed: actual-incumbent cutoff cancels challengers in
both full-frame and automatic buffering while retaining the exhaustive minimum
size and exact samples through both official decoders. The faulty fixture was
not treated as evidence that cancellation had run.

Reproduction for coefficient-derived original JPEG bytes:

```powershell
$env:RS_JXL_JPEG_INPUTS='../jxl_python/test_img;../jxl_python/test_img/mt_lay/t2;../jxl_python/test_img/mt_lay/t3'
cargo test --locked -p rs-jxl --test jpeg_reconstruction -- --ignored --nocapture --test-threads=1
```

`tests/jpeg_reconstruction.rs` is bounded to 17 explicitly selected JPEGs and
16 MiB per input. The independent coefficient test additionally requires
`RS_JXL_JPEG_COEFFICIENT_ORACLE` and is separately documented in JPEG_TOOL_PORT_STATUS.md.
Neither ordinary Cargo tests nor these JPEG tests imply a completed JXL archival
encoder or production acceptance of the unfinished pixel migration.

## Reconstruction Metadata And MA Checkpoint

The pure-Rust reconstruction serializer and Brotli suffix were tested independently
by replacing only `jbrd` in official coefficient-frame fixtures. All 17 JPEGs
passed with Brotli qualities 3 and 9 and official djxl 0.12 and 0.13: 68 exact
original-JPEG comparisons. The original entropy payloads were erased before
Rust serialization. This tests metadata compatibility, not a complete Rust
JPEG-to-JXL encoder. One fixture produced 536 bytes at Brotli 3 and 539 at
Brotli 9; the cumulative source quality schedule correctly keeps actual winners
rather than assuming higher quality is smaller.

The following debug workspace checkpoint passed 973 core tests (two opt-in
tests separate), 10 API tests, 14 pixel acceptance tests, 35 CLI tests and
ordinary integration helpers/doctest. Strict all-target Clippy passed. This
includes 13 metadata serializer tests, 3 Brotli tests, 9 JPEG layout tests,
17 MA scorer tests, packed MA property columns, and the unaligned embedded
Modular writer. Subsequent JPEG coefficient/order/quantizer work needs its own
validation; these numbers are not a claim about later files.

A release checkpoint after MA byte-column/FMA integration passed 47 official
decoder suites: 17 main, 16 float, 3 frame/metadata and 11 scheduler. Both official
versions checked original decoded samples, not compressed-byte identity.
That release build preceded the embedded Modular writer refactor; its debug
workspace checks above do not replace a new official decode run for that edit.

```powershell
$env:RS_JXL_CJXL='path/to/official/cjxl.exe'
$env:RS_JXL_DJXL='path/to/0.12/djxl.exe;path/to/0.13/djxl.exe'
$env:RS_JXL_JPEG_INPUTS='../jxl_python/test_img;../jxl_python/test_img/mt_lay/t2;../jxl_python/test_img/mt_lay/t3'
cargo test --locked -p rs-jxl --test jpeg_metadata_reference -- --include-ignored --nocapture
```

The metadata fixture uses nonrecursive discovery, 1..17 explicit JPEGs, 16 MiB
input/64 MiB output caps, child deadlines with kill/reap, and private checked
temporary files. It never substitutes the official encoder as a Rust backend.
